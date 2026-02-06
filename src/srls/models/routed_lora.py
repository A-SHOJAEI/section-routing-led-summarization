from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from srls.models.routing_state import RoutingState


@dataclass(frozen=True)
class LoRAParams:
    rank: int
    alpha: int
    dropout: float

    @property
    def scaling(self) -> float:
        return float(self.alpha) / float(self.rank)


class RoutedLoRALinear(nn.Module):
    """
    A lightweight LoRA wrapper that supports either:
    - shared LoRA (num_experts=1)
    - routed LoRA (num_experts=num_roles), selecting expert per token role id.

    This module is designed for encoder projections with inputs shaped (B, T, D).
    """

    def __init__(
        self,
        base: nn.Linear,
        *,
        state: RoutingState,
        num_experts: int,
        lora: LoRAParams,
        default_expert: int = 0,
    ) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"base must be nn.Linear, got {type(base)}")
        if num_experts < 1:
            raise ValueError("num_experts must be >= 1")
        if not (0 <= default_expert < num_experts):
            raise ValueError("default_expert out of range")

        self.base = base
        self.state = state
        self.num_experts = num_experts
        self.lora = lora
        self.default_expert = default_expert

        in_f = base.in_features
        out_f = base.out_features
        r = lora.rank

        # LoRA weights: A (down) and B (up) for each expert.
        self.A = nn.Parameter(torch.empty(num_experts, r, in_f))
        self.B = nn.Parameter(torch.empty(num_experts, out_f, r))
        self.dropout = nn.Dropout(p=lora.dropout)

        # Init per LoRA paper: A ~ N(0, 0.02), B = 0.
        nn.init.normal_(self.A, mean=0.0, std=0.02)
        nn.init.zeros_(self.B)

        # Freeze base linear by default (parameter-efficient fine-tuning).
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

    @torch.no_grad()
    def extra_repr(self) -> str:
        return f"num_experts={self.num_experts}, rank={self.lora.rank}, alpha={self.lora.alpha}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base projection.
        y = self.base(x)

        # LoRA adaptation.
        # Expected x shape: (B, T, Din) or (N, Din).
        if x.dim() == 2:
            # Treat as a single expert (no routing) for non-sequence inputs.
            x2 = self.dropout(x)
            a = self.A[0]
            b = self.B[0]
            l = F.linear(F.linear(x2, a), b) * self.lora.scaling
            return y + l

        if x.dim() != 3:
            raise ValueError(f"Expected x dim 2 or 3, got shape {tuple(x.shape)}")

        B, T, _ = x.shape
        role_ids = self.state.role_ids
        if role_ids is None:
            role_ids = torch.full((B, T), self.default_expert, device=x.device, dtype=torch.long)
        else:
            # HF models sometimes pass projections as (seq_len, batch, dim) and may internally pad
            # seq_len to a multiple of attention_window. Align (potentially unpadded) role_ids to
            # the first two dims of x, allowing for transpose + padding.
            if not torch.is_tensor(role_ids):
                role_ids = torch.tensor(role_ids, dtype=torch.long)
            role_ids = role_ids.to(device=x.device)

            if role_ids.shape == (B, T):
                pass
            elif role_ids.shape == (T, B):
                role_ids = role_ids.t()
            elif role_ids.shape[0] == B and role_ids.shape[1] <= T:
                # (B, t_small) -> pad time dim.
                pad = T - role_ids.shape[1]
                if pad:
                    role_ids = torch.cat(
                        [role_ids, torch.full((B, pad), self.default_expert, device=x.device, dtype=torch.long)],
                        dim=1,
                    )
            elif role_ids.shape[0] == B and role_ids.shape[1] >= T:
                # (B, t_big) -> truncate time dim (can happen for global-attn subproblems).
                role_ids = role_ids[:, :T]
            elif role_ids.shape[0] == T and role_ids.shape[1] <= B:
                # Common case for (seq_len, batch, dim) projections: role_ids is (batch, seq_unpadded).
                # Transpose to (seq_unpadded, batch) and pad seq_len.
                role_ids = role_ids.t()
                pad = B - role_ids.shape[0]
                if pad:
                    role_ids = torch.cat(
                        [role_ids, torch.full((pad, T), self.default_expert, device=x.device, dtype=torch.long)],
                        dim=0,
                    )
            elif role_ids.shape[0] == T and role_ids.shape[1] >= B:
                # (T, b_big) -> transpose then truncate/pad.
                role_ids = role_ids.t()
                role_ids = role_ids[:B, :]
            else:
                # Fallback: if we can't safely align (e.g. internal sub-attention shapes),
                # disable routing for this projection call rather than crashing.
                role_ids = torch.full((B, T), self.default_expert, device=x.device, dtype=torch.long)

        x_flat = self.dropout(x).reshape(B * T, -1)
        r_flat = role_ids.reshape(B * T)

        # Compute LoRA for each expert on its token subset to avoid dense (B,T,E) tensors.
        lora_out = torch.zeros((B * T, self.base.out_features), device=x.device, dtype=y.dtype)
        for e in range(self.num_experts):
            idx = (r_flat == e).nonzero(as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                continue
            a = self.A[e]
            b = self.B[e]
            xe = x_flat.index_select(0, idx)
            le = F.linear(F.linear(xe, a), b) * self.lora.scaling
            lora_out.index_copy_(0, idx, le)

        return y + lora_out.reshape(B, T, -1)
