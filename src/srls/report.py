from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from srls.config import cfg_get, get_run_paths, load_config
from srls.utils.hashing import atomic_write_text, read_json


RUNS = [
    ("baseline", "baseline_shared_lora_no_section_tokens"),
    ("main", "main_routed_lora_section_tokens_aux_loss"),
    ("ablation_no_routing", "ablation_no_routing_section_tokens_shared_lora"),
]


def _get_metric(res: dict[str, Any], key: str) -> tuple[float, float, float]:
    m = res["metrics"][key]
    mean = float(m["mean"])
    lo, hi = m["ci95"]
    return mean, float(lo), float(hi)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    output_root = Path(cfg_get(cfg, "train.output_dir", "outputs"))
    out_path = Path("report.md")

    rows = []
    missing = []
    for run_type, run_name in RUNS:
        paths = get_run_paths(output_root, run_name)
        if not paths.results_json.exists():
            missing.append((run_type, str(paths.results_json)))
            continue
        res = read_json(paths.results_json)
        r1, r1l, r1h = _get_metric(res, "rouge1_f")
        r2, r2l, r2h = _get_metric(res, "rouge2_f")
        rl, rll, rlh = _get_metric(res, "rougeL_f")
        bs, bsl, bsh = _get_metric(res, "bertscore_f1")
        nli, nlil, nlih = _get_metric(res, "nli_entail_minus_contra")
        rows.append(
            {
                "run_type": run_type,
                "run_name": run_name,
                "n": int(res.get("num_examples", 0)),
                "rouge1": (r1, r1l, r1h),
                "rouge2": (r2, r2l, r2h),
                "rougeL": (rl, rll, rlh),
                "bertscore_f1": (bs, bsl, bsh),
                "nli": (nli, nlil, nlih),
            }
        )

    if missing:
        msg = "\n".join([f"- {rt}: missing {p}" for rt, p in missing])
        raise FileNotFoundError(f"Missing evaluation outputs:\n{msg}")

    def fmt_trip(t: tuple[float, float, float]) -> str:
        m, lo, hi = t
        return f"{m:.4f} [{lo:.4f}, {hi:.4f}]"

    lines = []
    lines.append("# Report: Section-Routed LED Summarization\n")
    lines.append(f"Config: `{args.config}`\n")
    lines.append("## Summary Table (Mean [95% bootstrap CI])\n")
    lines.append("| Run | N | ROUGE-1 F | ROUGE-2 F | ROUGE-L F | BERTScore F1 | NLI (ent-contr) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            "| {run_type} | {n} | {r1} | {r2} | {rl} | {bs} | {nli} |".format(
                run_type=r["run_type"],
                n=r["n"],
                r1=fmt_trip(r["rouge1"]),
                r2=fmt_trip(r["rouge2"]),
                rl=fmt_trip(r["rougeL"]),
                bs=fmt_trip(r["bertscore_f1"]),
                nli=fmt_trip(r["nli"]),
            )
        )

    lines.append("\n## Implemented Comparisons\n")
    lines.append("- Baseline: no section tokens + single shared LoRA (no routing).")
    lines.append("- Main: section tokens + routed LoRA (per-role expert) + auxiliary role classification loss.")
    lines.append("- Ablation (No routing): section tokens kept + single shared LoRA (routing disabled).\n")

    atomic_write_text(out_path, "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

