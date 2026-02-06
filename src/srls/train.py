from __future__ import annotations

import argparse
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from srls.checkpoint import save_checkpoint
from srls.config import cfg_get, get_run_paths, load_config
from srls.models.build import BuildSpec, build_model, build_tokenizer
from srls.utils.hashing import write_json
from srls.utils.repro import set_reproducibility, set_torch_reproducibility


def _pad_2d(seqs: list[list[int]], pad: int, max_len: int) -> torch.Tensor:
    out = torch.full((len(seqs), max_len), pad, dtype=torch.long)
    for i, s in enumerate(seqs):
        s = s[:max_len]
        out[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    return out


def make_collate_fn(tokenizer, *, label_pad_id: int = -100, role_pad_id: int = 0):
    pad_id = int(tokenizer.pad_token_id) if tokenizer.pad_token_id is not None else int(tokenizer.eos_token_id)

    def collate(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_in = max(len(x["input_ids"]) for x in batch)
        max_lab = max(len(x["labels"]) for x in batch)

        input_ids = _pad_2d([x["input_ids"] for x in batch], pad_id, max_in)
        attention_mask = _pad_2d([x["attention_mask"] for x in batch], 0, max_in)
        global_attention_mask = _pad_2d([x["global_attention_mask"] for x in batch], 0, max_in)
        role_ids = _pad_2d([x["role_ids"] for x in batch], role_pad_id, max_in)

        labels = _pad_2d([x["labels"] for x in batch], pad_id, max_lab)
        labels = labels.masked_fill(labels == pad_id, label_pad_id)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "global_attention_mask": global_attention_mask,
            "role_ids": role_ids,
            "labels": labels,
        }

    return collate


def freeze_all_params(model: torch.nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad_(False)


def unfreeze_lora_and_aux(model: torch.nn.Module) -> int:
    n = 0
    for name, p in model.named_parameters():
        if name.endswith(".A") or name.endswith(".B") or "role_head" in name:
            p.requires_grad_(True)
            n += p.numel()
    return n


def get_run_spec(cfg: dict[str, Any], run: str) -> tuple[str, str, BuildSpec]:
    roles = list(cfg_get(cfg, "model.section_roles"))
    base_model_name = str(cfg_get(cfg, "model.base_model_name"))
    lora_rank = int(cfg_get(cfg, "model.lora_rank"))
    lora_alpha = int(cfg_get(cfg, "model.lora_alpha"))
    lora_dropout = float(cfg_get(cfg, "model.lora_dropout"))
    target_modules = list(cfg_get(cfg, "model.target_modules"))
    aux_w = float(cfg_get(cfg, "model.aux_loss_weight", 0.0))

    if run == "baseline":
        return (
            "baseline_shared_lora_no_section_tokens",
            "baseline_no_section_tokens",
            BuildSpec(
                base_model_name=base_model_name,
                roles=roles,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules,
                routed=False,
                aux_loss_weight=0.0,  # no section markers => no discourse supervision
            ),
        )
    if run == "main":
        return (
            "main_routed_lora_section_tokens_aux_loss",
            "with_section_tokens",
            BuildSpec(
                base_model_name=base_model_name,
                roles=roles,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules,
                routed=True,
                aux_loss_weight=aux_w,
            ),
        )
    if run == "ablation_no_routing":
        # Ablation exactly as described: keep section tokens but disable routing (single shared LoRA).
        return (
            "ablation_no_routing_section_tokens_shared_lora",
            "with_section_tokens",
            BuildSpec(
                base_model_name=base_model_name,
                roles=roles,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules,
                routed=False,
                aux_loss_weight=aux_w,
            ),
        )
    raise ValueError(f"Unknown run: {run}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run", required=True, choices=["baseline", "main", "ablation_no_routing"])
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg_get(cfg, "project.seed", 7))
    deterministic = bool(cfg_get(cfg, "project.deterministic", False))
    set_reproducibility(seed, deterministic=deterministic)
    set_torch_reproducibility(seed, deterministic=deterministic)

    mixed_precision = str(cfg_get(cfg, "train.mixed_precision", "no"))
    accelerator = Accelerator(mixed_precision=None if mixed_precision == "no" else mixed_precision)

    processed_dir = Path(cfg_get(cfg, "data.processed_dir"))
    cache_dir = str(cfg_get(cfg, "data.cache_dir", "data/hf_cache"))
    output_root = Path(cfg_get(cfg, "train.output_dir", "outputs"))
    output_root.mkdir(parents=True, exist_ok=True)

    run_name, variant_dirname, spec = get_run_spec(cfg, args.run)
    paths = get_run_paths(output_root, run_name)
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    paths.model_dir.mkdir(parents=True, exist_ok=True)

    # Build tokenizer (must match preprocessing special tokens).
    tok_path = processed_dir / "tokenizer"
    if tok_path.exists():
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(tok_path), use_fast=True)
    else:
        tokenizer = build_tokenizer(spec, cache_dir=cache_dir)

    # Load processed dataset variant.
    ds_path = processed_dir / variant_dirname
    if not ds_path.exists():
        raise FileNotFoundError(f"Processed dataset not found: {ds_path}. Run `make data` first.")
    dsd = load_from_disk(str(ds_path))
    train_ds = dsd["train"]
    valid_ds = dsd["validation"]

    # Model.
    model = build_model(spec, tokenizer, cache_dir=cache_dir)
    freeze_all_params(model)
    trainable = unfreeze_lora_and_aux(model)
    if trainable == 0:
        raise RuntimeError("No trainable parameters found (LoRA/Aux).")

    if bool(cfg_get(cfg, "train.grad_checkpointing", False)) and hasattr(model.base, "gradient_checkpointing_enable"):
        model.base.gradient_checkpointing_enable()
        # When doing parameter-efficient tuning with most weights frozen, re-entrant gradient
        # checkpointing can drop grads unless at least one input to the checkpointed blocks
        # requires grad. Transformers provides this helper to mark embedding outputs accordingly.
        if hasattr(model.base, "enable_input_require_grads"):
            model.base.enable_input_require_grads()

    collate = make_collate_fn(tokenizer, role_pad_id=spec.roles.index("other"))
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg_get(cfg, "train.per_device_train_batch_size", 1)),
        shuffle=True,
        collate_fn=collate,
        generator=torch.Generator().manual_seed(seed),
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=int(cfg_get(cfg, "train.per_device_eval_batch_size", 1)),
        shuffle=False,
        collate_fn=collate,
    )

    lr = float(cfg_get(cfg, "train.learning_rate"))
    wd = float(cfg_get(cfg, "train.weight_decay", 0.0))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=wd)

    max_steps = int(cfg_get(cfg, "train.max_steps"))
    warmup_steps = int(cfg_get(cfg, "train.warmup_steps", 0))
    from transformers import get_linear_schedule_with_warmup

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=max_steps)

    grad_acc = int(cfg_get(cfg, "train.gradient_accumulation_steps", 1))

    model, optimizer, train_loader, valid_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, valid_loader, scheduler
    )

    log_every = int(cfg_get(cfg, "train.log_every_steps", 10))
    eval_every = int(cfg_get(cfg, "train.eval_every_steps", max_steps))
    save_every = int(cfg_get(cfg, "train.save_every_steps", max_steps))

    start = time.time()
    step = 0
    running_loss = 0.0
    running_tokens = 0
    pbar = tqdm(total=max_steps, disable=not accelerator.is_local_main_process, desc=f"train:{run_name}")

    model.train()
    train_iter = iter(train_loader)
    last_log = time.time()

    while step < max_steps:
        optimizer.zero_grad(set_to_none=True)
        for _ in range(grad_acc):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            out = model(**batch)
            loss = out.loss / float(grad_acc)
            accelerator.backward(loss)

            # Token count proxy for throughput.
            running_tokens += int(batch["attention_mask"].sum().item()) + int((batch["labels"] != -100).sum().item())
            running_loss += float((loss.detach().item()) * float(grad_acc))

        optimizer.step()
        scheduler.step()
        step += 1
        pbar.update(1)

        if step % log_every == 0 and accelerator.is_local_main_process:
            now = time.time()
            dt = max(now - last_log, 1e-9)
            toks_per_s = running_tokens / dt
            avg_loss = running_loss / log_every
            pbar.set_postfix({"loss": f"{avg_loss:.4f}", "tok/s": f"{toks_per_s:.1f}"})
            running_loss = 0.0
            running_tokens = 0
            last_log = now

        if step % eval_every == 0:
            model.eval()
            losses = []
            with torch.no_grad():
                for vb in valid_loader:
                    o = model(**vb)
                    losses.append(accelerator.gather_for_metrics(o.loss.detach()).float().cpu())
            if accelerator.is_local_main_process:
                vloss = torch.stack(losses).mean().item() if losses else float("nan")
                write_json(paths.run_dir / "valid_metrics.json", {"step": step, "loss": vloss})
            model.train()

        if step % save_every == 0 and accelerator.is_local_main_process:
            unwrapped = accelerator.unwrap_model(model)
            save_checkpoint(
                run_name=run_name,
                model=unwrapped,
                tokenizer=tokenizer,
                spec=spec,
                cfg=cfg,
                out_model_pt=paths.model_pt,
                out_tokenizer_dir=paths.tokenizer_dir,
            )

    pbar.close()
    end = time.time()

    metrics = {
        "run_name": run_name,
        "run_type": args.run,
        "spec": asdict(spec),
        "seed": seed,
        "deterministic": deterministic,
        "mixed_precision": mixed_precision,
        "steps": step,
        "wall_time_sec": end - start,
    }
    if torch.cuda.is_available():
        metrics["cuda"] = {
            "device_count": torch.cuda.device_count(),
            "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        }

    if accelerator.is_local_main_process:
        write_json(paths.train_metrics_json, metrics)
        # Ensure a final checkpoint is present.
        unwrapped = accelerator.unwrap_model(model)
        save_checkpoint(
            run_name=run_name,
            model=unwrapped,
            tokenizer=tokenizer,
            spec=spec,
            cfg=cfg,
            out_model_pt=paths.model_pt,
            out_tokenizer_dir=paths.tokenizer_dir,
        )


if __name__ == "__main__":
    # Reduce tokenizer parallelism nondeterminism/noise.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
