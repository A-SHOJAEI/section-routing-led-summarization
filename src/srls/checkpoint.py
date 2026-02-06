from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from srls.models.build import BuildSpec, build_model


@dataclass(frozen=True)
class CheckpointMeta:
    run_name: str
    spec: dict[str, Any]
    config: dict[str, Any]
    torch_version: str


def save_checkpoint(
    *,
    run_name: str,
    model,
    tokenizer,
    spec: BuildSpec,
    cfg: dict[str, Any],
    out_model_pt: str | Path,
    out_tokenizer_dir: str | Path,
) -> None:
    out_model_pt = Path(out_model_pt)
    out_model_pt.parent.mkdir(parents=True, exist_ok=True)
    Path(out_tokenizer_dir).mkdir(parents=True, exist_ok=True)

    tokenizer.save_pretrained(str(out_tokenizer_dir))

    meta = CheckpointMeta(
        run_name=run_name,
        spec=asdict(spec),
        config=cfg,
        torch_version=torch.__version__,
    )
    payload = {
        "meta": asdict(meta),
        "model_state_dict": model.state_dict(),
    }
    torch.save(payload, str(out_model_pt))


def load_checkpoint(
    model_pt: str | Path,
    *,
    cache_dir: str | None = None,
    map_location: str | torch.device | None = "cpu",
) -> tuple[Any, dict[str, Any]]:
    """
    Returns: (model, meta)
    Tokenizer is loaded separately from meta['tokenizer_dir'] or by caller.
    """
    # We only load checkpoints produced by this repo, but newer PyTorch versions
    # default to `weights_only=True` which can reject harmless metadata in pickles.
    # Explicitly allow full load for our own files.
    try:
        payload = torch.load(str(model_pt), map_location=map_location, weights_only=False)
    except TypeError:
        payload = torch.load(str(model_pt), map_location=map_location)
    meta = payload["meta"]
    spec = BuildSpec(**meta["spec"])
    # Tokenizer must match the checkpoint; infer it from spec and caller config.
    from transformers import AutoTokenizer

    # Prefer tokenizer saved alongside model if present.
    model_dir = Path(model_pt).parent
    tok_dir = model_dir / "tokenizer"
    if tok_dir.exists():
        tokenizer = AutoTokenizer.from_pretrained(str(tok_dir), use_fast=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(spec.base_model_name, cache_dir=cache_dir, use_fast=True)
        tokenizer.add_special_tokens({"additional_special_tokens": [f"<sec:{r}>" for r in spec.roles]})

    model = build_model(spec, tokenizer, cache_dir=cache_dir)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return (model, {"meta": meta, "tokenizer": tokenizer})
