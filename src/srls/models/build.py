from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch.nn as nn
from transformers import AutoTokenizer, LEDForConditionalGeneration

from srls.data.roles import role_special_tokens
from srls.models.led_wrapper import AuxLossConfig, LEDWithRoutingAndAuxLoss
from srls.models.routed_lora import LoRAParams, RoutedLoRALinear
from srls.models.routing_state import RoutingState


@dataclass(frozen=True)
class BuildSpec:
    base_model_name: str
    roles: list[str]
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    target_modules: list[str]
    routed: bool
    aux_loss_weight: float


def _iter_named_linears(model: nn.Module) -> Iterable[tuple[str, nn.Linear]]:
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            yield name, mod


def _set_module(root: nn.Module, name: str, new_mod: nn.Module) -> None:
    parts = name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_mod)


def build_tokenizer(spec: BuildSpec, cache_dir: str | None = None):
    tok = AutoTokenizer.from_pretrained(spec.base_model_name, cache_dir=cache_dir, use_fast=True)
    tok.add_special_tokens({"additional_special_tokens": role_special_tokens(spec.roles)})
    return tok


def build_model(spec: BuildSpec, tokenizer, cache_dir: str | None = None) -> nn.Module:
    base = LEDForConditionalGeneration.from_pretrained(spec.base_model_name, cache_dir=cache_dir)
    base.resize_token_embeddings(len(tokenizer))

    # Routed LoRA only on the encoder (plan: shared decoder).
    state = RoutingState()
    lora = LoRAParams(rank=int(spec.lora_rank), alpha=int(spec.lora_alpha), dropout=float(spec.lora_dropout))
    num_experts = len(spec.roles) if spec.routed else 1
    default_expert = spec.roles.index("other") if num_experts > 1 else 0

    # The plan names common Transformer projection modules (q_proj/v_proj), but LED's Longformer
    # encoder uses `query`/`value` naming. Map to support both conventions in configs.
    targets = set(spec.target_modules)
    if "q_proj" in targets:
        targets.add("query")
    if "k_proj" in targets:
        targets.add("key")
    if "v_proj" in targets:
        targets.add("value")
    if "out_proj" in targets:
        targets.add("output")

    targets |= {t + "_global" for t in list(targets)}

    sec_token_ids = [tokenizer.convert_tokens_to_ids(t) for t in role_special_tokens(spec.roles)]

    for name, linear in list(_iter_named_linears(base)):
        if ".encoder." not in name:
            continue
        leaf = name.split(".")[-1]
        if leaf not in targets:
            continue
        wrapped = RoutedLoRALinear(linear, state=state, num_experts=num_experts, lora=lora, default_expert=default_expert)
        _set_module(base, name, wrapped)

    aux = AuxLossConfig(
        enabled=spec.aux_loss_weight > 0,
        weight=float(spec.aux_loss_weight),
        sec_token_ids=sec_token_ids,
        num_roles=len(spec.roles),
    )
    return LEDWithRoutingAndAuxLoss(base, state=state, aux=aux)
