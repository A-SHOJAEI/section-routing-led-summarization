from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RoutingState:
    """
    Mutable per-forward state shared by routed layers.

    We store the per-token role ids here so routed LoRA layers can read them
    without changing the underlying Transformers attention module signatures.
    """

    role_ids: torch.Tensor | None = None  # (batch, seq)

