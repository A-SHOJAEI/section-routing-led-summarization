from __future__ import annotations

import argparse
import os
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from accelerate import Accelerator
from datasets import load_from_disk
from rouge_score import rouge_scorer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from srls.checkpoint import load_checkpoint
from srls.config import cfg_get, get_run_paths, load_config
from srls.train import get_run_spec, make_collate_fn
from srls.utils.hashing import atomic_write_text, write_json


def _bootstrap_mean_ci(values: np.ndarray, samples: int, seed: int = 0) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    n = values.shape[0]
    mean = float(values.mean()) if n else float("nan")
    if n == 0 or samples <= 0:
        return {"mean": mean, "ci95": [float("nan"), float("nan")]}

    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for i in range(samples):
        idx = rng.integers(0, n, size=n)
        means[i] = values[idx].mean()
    lo, hi = np.percentile(means, [2.5, 97.5]).tolist()
    return {"mean": mean, "ci95": [float(lo), float(hi)]}


def _nli_label_ids(model) -> tuple[int, int]:
    # Try to locate label ids robustly.
    label2id = getattr(model.config, "label2id", {}) or {}
    inv = {str(k).lower(): int(v) for k, v in label2id.items()}
    for k in list(inv.keys()):
        if k.startswith("label_"):
            # Some configs expose {"LABEL_0": 0, ...}
            pass

    def find(keys: list[str], fallback: int) -> int:
        for k in keys:
            if k in inv:
                return inv[k]
            # Common variants.
            if k.upper() in label2id:
                return int(label2id[k.upper()])
        return fallback

    # Common MNLI ordering is (contradiction=0, neutral=1, entailment=2).
    ent = find(["entailment"], 2)
    con = find(["contradiction"], 0)
    return ent, con


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run", required=True, choices=["baseline", "main", "ablation_no_routing"])
    args = ap.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    cfg = load_config(args.config)
    processed_dir = Path(cfg_get(cfg, "data.processed_dir"))
    output_root = Path(cfg_get(cfg, "train.output_dir", "outputs"))
    cache_dir = str(cfg_get(cfg, "data.cache_dir", "data/hf_cache"))

    run_name, variant_dirname, _ = get_run_spec(cfg, args.run)
    paths = get_run_paths(output_root, run_name)
    paths.eval_dir.mkdir(parents=True, exist_ok=True)

    if not paths.model_pt.exists():
        raise FileNotFoundError(f"Missing checkpoint: {paths.model_pt}. Run training first.")

    model, ck = load_checkpoint(paths.model_pt, cache_dir=cache_dir, map_location="cpu")
    tokenizer = ck["tokenizer"]
    model.eval()

    ds_path = processed_dir / variant_dirname
    dsd = load_from_disk(str(ds_path))
    test_ds = dsd["test"]

    collate = make_collate_fn(tokenizer, role_pad_id=int(cfg_get(cfg, "model.section_roles").index("other")))

    def collate_with_text(batch: list[dict[str, Any]]) -> dict[str, Any]:
        tens = collate(batch)
        tens["source_text"] = [x["source_text"] for x in batch]
        tens["reference_summary"] = [x["reference_summary"] for x in batch]
        return tens

    loader = DataLoader(
        test_ds,
        batch_size=int(cfg_get(cfg, "train.per_device_eval_batch_size", 1)),
        shuffle=False,
        collate_fn=collate_with_text,
    )

    accelerator = Accelerator()
    model, loader = accelerator.prepare(model, loader)
    unwrap = accelerator.unwrap_model(model)

    max_new = int(cfg_get(cfg, "eval.max_generate_tokens", 128))
    num_beams = int(cfg_get(cfg, "eval.num_beams", 2))

    preds: list[str] = []
    refs: list[str] = []
    sources: list[str] = []

    gen_start = time.time()
    for batch in tqdm(loader, disable=not accelerator.is_local_main_process, desc=f"gen:{run_name}"):
        # Ensure routed layers see the per-token roles during generation.
        if "role_ids" in batch:
            unwrap.state.role_ids = batch["role_ids"]
        with torch.no_grad():
            gen_ids = unwrap.base.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                global_attention_mask=batch["global_attention_mask"],
                max_new_tokens=max_new,
                num_beams=num_beams,
            )
        # Gather across processes.
        gen_ids = accelerator.gather_for_metrics(gen_ids).cpu().numpy().tolist()
        pred_text = tokenizer.batch_decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        preds.extend([t.strip() for t in pred_text])

        # Text fields are only present on local process; rebuild via gather is overkill for smoke.
        if accelerator.is_local_main_process:
            refs.extend([t.strip() for t in batch["reference_summary"]])
            sources.extend([t for t in batch["source_text"]])

    if not accelerator.is_local_main_process:
        return

    gen_end = time.time()

    # ROUGE (per-example, f-measure).
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1 = np.array([scorer.score(r, p)["rouge1"].fmeasure for p, r in zip(preds, refs, strict=True)], dtype=np.float64)
    r2 = np.array([scorer.score(r, p)["rouge2"].fmeasure for p, r in zip(preds, refs, strict=True)], dtype=np.float64)
    rl = np.array([scorer.score(r, p)["rougeL"].fmeasure for p, r in zip(preds, refs, strict=True)], dtype=np.float64)

    # BERTScore (per-example).
    from bert_score import score as bert_score

    bs_model = str(cfg_get(cfg, "eval.bertscore_model", "distilroberta-base"))
    P, R, F1 = bert_score(preds, refs, model_type=bs_model, lang="en", verbose=False, device="cuda" if torch.cuda.is_available() else "cpu")
    bs_f1 = F1.detach().cpu().numpy().astype(np.float64)

    # NLI factuality proxy: entailment - contradiction (higher is better).
    nli_model_name = str(cfg_get(cfg, "eval.nli_model"))
    nli_tok = AutoTokenizer.from_pretrained(nli_model_name, cache_dir=cache_dir, use_fast=True)
    nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name, cache_dir=cache_dir)
    nli_model.eval()
    nli_model.to("cuda" if torch.cuda.is_available() else "cpu")
    ent_id, con_id = _nli_label_ids(nli_model)

    max_chars = int(cfg_get(cfg, "eval.max_source_chars_for_nli", 6000))
    nli_scores = []
    bs = 8
    for i in tqdm(range(0, len(preds), bs), desc="nli", leave=False):
        src_b = [s[:max_chars] for s in sources[i : i + bs]]
        hyp_b = preds[i : i + bs]
        enc = nli_tok(
            src_b,
            hyp_b,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        enc = {k: v.to(nli_model.device) for k, v in enc.items()}
        with torch.no_grad():
            logits = nli_model(**enc).logits
            probs = torch.softmax(logits, dim=-1)
            score = (probs[:, ent_id] - probs[:, con_id]).detach().cpu().numpy().tolist()
            nli_scores.extend(score)
    nli_scores = np.array(nli_scores, dtype=np.float64)

    boot = int(cfg_get(cfg, "eval.bootstrap_samples", 200))
    seed = int(cfg_get(cfg, "project.seed", 7))

    metrics = {
        "rouge1_f": _bootstrap_mean_ci(r1, boot, seed=seed),
        "rouge2_f": _bootstrap_mean_ci(r2, boot, seed=seed),
        "rougeL_f": _bootstrap_mean_ci(rl, boot, seed=seed),
        "bertscore_f1": _bootstrap_mean_ci(bs_f1, boot, seed=seed),
        "nli_entail_minus_contra": _bootstrap_mean_ci(nli_scores, boot, seed=seed),
    }

    # Save predictions (for error analysis) and results JSON.
    pred_path = paths.eval_dir / "predictions.jsonl"
    lines = []
    for p, r, s in zip(preds, refs, sources, strict=True):
        obj = {"prediction": p, "reference": r, "source_text": s[:4000]}
        lines.append(obj)
    atomic_write_text(pred_path, "\n".join([json_dumps(o) for o in lines]) + "\n")

    results = {
        "run_name": run_name,
        "run_type": args.run,
        "num_examples": len(preds),
        "generation": {"max_new_tokens": max_new, "num_beams": num_beams, "wall_time_sec": gen_end - gen_start},
        "metrics": metrics,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
        },
        "config_path": str(args.config),
    }
    if torch.cuda.is_available():
        results["environment"]["cuda_device_count"] = torch.cuda.device_count()
        results["environment"]["cuda_name_0"] = torch.cuda.get_device_name(0)

    write_json(paths.results_json, results)


def json_dumps(obj: Any) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False)


if __name__ == "__main__":
    main()
