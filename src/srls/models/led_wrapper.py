from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from srls.models.routing_state import RoutingState


@dataclass(frozen=True)
class AuxLossConfig:
    enabled: bool
    weight: float
    sec_token_ids: list[int]
    num_roles: int


class LEDWithRoutingAndAuxLoss(nn.Module):
    """
    Wraps LEDForConditionalGeneration to:
    - set RoutingState.role_ids per forward (used by routed LoRA layers)
    - optionally add an auxiliary role-classification loss at section marker tokens
    """

    def __init__(self, base_model: nn.Module, *, state: RoutingState, aux: AuxLossConfig) -> None:
        super().__init__()
        self.base = base_model
        self.state = state
        self.aux = aux

        hidden = getattr(self.base.config, "d_model", None)
        if hidden is None:
            raise ValueError("LED config missing d_model")

        self.role_head = nn.Linear(int(hidden), int(aux.num_roles)) if aux.enabled and aux.weight > 0 else None

    def forward(self, **kwargs: Any):
        role_ids = kwargs.pop("role_ids", None)
        if role_ids is not None and not torch.is_tensor(role_ids):
            role_ids = torch.tensor(role_ids, dtype=torch.long)

        self.state.role_ids = role_ids

        # Always return encoder hidden state if we need aux loss.
        if self.role_head is not None:
            kwargs["output_hidden_states"] = False
            kwargs["return_dict"] = True

        out = self.base(**kwargs)

        if self.role_head is None:
            return out

        input_ids = kwargs.get("input_ids", None)
        if input_ids is None:
            # If callers used inputs_embeds, we can't locate section markers reliably.
            return out

        enc = out.encoder_last_hidden_state  # (B, T, H)
        B, T, _ = enc.shape
        if role_ids is None or role_ids.shape != (B, T):
            return out

        sec_ids = torch.tensor(self.aux.sec_token_ids, device=input_ids.device, dtype=input_ids.dtype)
        sec_mask = (input_ids[..., None] == sec_ids[None, None, :]).any(dim=-1)  # (B, T)
        if sec_mask.sum().item() == 0:
            return out

        reps = enc[sec_mask]  # (Nsec, H)
        labels = role_ids[sec_mask]  # (Nsec,)
        logits = self.role_head(reps)
        aux_loss = nn.functional.cross_entropy(logits, labels)
        out.loss = out.loss + (float(self.aux.weight) * aux_loss)
        return out

