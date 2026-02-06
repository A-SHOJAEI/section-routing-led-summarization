from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping, got: {type(data)}")
    return data


def cfg_get(cfg: dict[str, Any], key_path: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in key_path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    model_dir: Path
    model_pt: Path
    tokenizer_dir: Path
    train_metrics_json: Path
    eval_dir: Path
    results_json: Path


def get_run_paths(output_root: str | Path, run_name: str) -> RunPaths:
    run_dir = Path(output_root) / run_name
    model_dir = run_dir / "model"
    eval_dir = run_dir / "eval"
    return RunPaths(
        run_dir=run_dir,
        model_dir=model_dir,
        model_pt=model_dir / "model.pt",
        tokenizer_dir=model_dir / "tokenizer",
        train_metrics_json=run_dir / "train_metrics.json",
        eval_dir=eval_dir,
        results_json=eval_dir / "results.json",
    )

