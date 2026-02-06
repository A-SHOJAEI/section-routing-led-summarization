from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer

from srls.config import cfg_get, load_config
from srls.data.roles import SectionRoleMapper, role_special_tokens
from srls.utils.hashing import compute_tree_checksums, write_json


def _iter_take(ds_iterable, n: int):
    return list(itertools.islice(ds_iterable, n))


def _get_text_fields(ex: dict[str, Any]) -> tuple[list[str], list[str], str]:
    """
    Returns: (section_names, sections, abstract)
    Falls back to 'article' if structured sections aren't present.
    """
    abstract = ex.get("abstract") or ""
    section_names = ex.get("section_names")
    sections = ex.get("sections")

    if isinstance(section_names, list) and isinstance(sections, list) and len(section_names) == len(sections):
        # Ensure strings.
        names = [("" if n is None else str(n)) for n in section_names]
        secs = [("" if s is None else str(s)) for s in sections]
        return names, secs, str(abstract)

    # Fallback to a single "article" field.
    article = ex.get("article") or ""
    return ["article"], [str(article)], str(abstract)


def build_source_text(
    section_names: list[str],
    sections: list[str],
    mapper: SectionRoleMapper,
    use_section_tokens: bool,
) -> str:
    parts: list[str] = []
    for name, text in zip(section_names, sections, strict=True):
        text = (text or "").strip()
        if not text:
            continue
        if use_section_tokens:
            role = mapper.map_name(name)
            parts.append(f"<sec:{role}>\n{text}")
        else:
            parts.append(text)
    return "\n\n".join(parts).strip()


def derive_role_ids(input_ids: list[int], sec_token_id_to_role_id: dict[int, int], default_role_id: int) -> list[int]:
    role_ids: list[int] = []
    cur_role = default_role_id
    for tid in input_ids:
        if tid in sec_token_id_to_role_id:
            cur_role = sec_token_id_to_role_id[tid]
        role_ids.append(cur_role)
    return role_ids


def derive_global_attention_mask(input_ids: list[int], global_token_ids: set[int]) -> list[int]:
    # LED typically uses global attention on <s>. We also set it on section marker tokens.
    return [1 if i == 0 or tid in global_token_ids else 0 for i, tid in enumerate(input_ids)]


def prepare_split(
    raw_split,
    *,
    tokenizer,
    mapper: SectionRoleMapper,
    roles: list[str],
    use_section_tokens: bool,
    max_input_tokens: int,
    max_target_tokens: int,
    max_examples: int | None,
) -> Dataset:
    if hasattr(raw_split, "__iter__") and not isinstance(raw_split, Dataset):
        if max_examples is None:
            raise ValueError("streaming=True requires max_*_examples caps per split")
        rows = _iter_take(raw_split, max_examples)
        raw = Dataset.from_list(rows)
    else:
        raw = raw_split
        if max_examples is not None:
            raw = raw.select(range(min(max_examples, len(raw))))

    sec_tokens = role_special_tokens(roles)
    sec_token_ids = {tokenizer.convert_tokens_to_ids(t) for t in sec_tokens}
    sec_token_id_to_role_id = {tokenizer.convert_tokens_to_ids(f"<sec:{r}>"): i for i, r in enumerate(roles)}
    default_role_id = roles.index("other")

    def _map_batch(batch: dict[str, list[Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {
            "input_ids": [],
            "attention_mask": [],
            "global_attention_mask": [],
            "labels": [],
            "role_ids": [],
            "source_text": [],
            "reference_summary": [],
        }
        for i in range(len(batch["abstract"])):
            ex = {k: batch[k][i] for k in batch.keys()}
            names, secs, abstract = _get_text_fields(ex)
            src = build_source_text(names, secs, mapper, use_section_tokens=use_section_tokens)
            tgt = abstract.strip()

            # Tokenize.
            enc = tokenizer(
                src,
                truncation=True,
                max_length=max_input_tokens,
                padding=False,
            )
            dec = tokenizer(
                text_target=tgt,
                truncation=True,
                max_length=max_target_tokens,
                padding=False,
            )

            input_ids = enc["input_ids"]
            attn = enc["attention_mask"]
            role_ids = derive_role_ids(input_ids, sec_token_id_to_role_id, default_role_id) if use_section_tokens else [default_role_id] * len(input_ids)
            gmask = derive_global_attention_mask(input_ids, sec_token_ids)

            out["input_ids"].append(input_ids)
            out["attention_mask"].append(attn)
            out["global_attention_mask"].append(gmask)
            out["labels"].append(dec["input_ids"])
            out["role_ids"].append(role_ids)
            # Keep raw text for evaluation/reporting; cap to keep disk sane.
            out["source_text"].append(src[:200_000])
            out["reference_summary"].append(tgt)
        return out

    cols = list(raw.column_names)
    # Ensure 'abstract' exists in batch.
    if "abstract" not in cols:
        raise KeyError(f"Dataset split is missing required column 'abstract'. Columns: {cols}")

    mapped = raw.map(_map_batch, batched=True, remove_columns=cols, desc="Tokenizing")
    return mapped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)

    dataset_name = cfg_get(cfg, "data.dataset_name")
    dataset_config = cfg_get(cfg, "data.dataset_config")
    streaming = bool(cfg_get(cfg, "data.streaming", False))
    trust_remote_code = bool(cfg_get(cfg, "data.trust_remote_code", False))
    cache_dir = cfg_get(cfg, "data.cache_dir", "data/hf_cache")
    processed_dir = Path(cfg_get(cfg, "data.processed_dir"))
    processed_dir.mkdir(parents=True, exist_ok=True)

    roles = list(cfg_get(cfg, "model.section_roles"))
    mapper = SectionRoleMapper(roles=roles)

    base_model_name = cfg_get(cfg, "model.base_model_name")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, cache_dir=cache_dir, use_fast=True)
    tokenizer.add_special_tokens({"additional_special_tokens": role_special_tokens(roles)})

    raw: DatasetDict = load_dataset(
        dataset_name,
        dataset_config,
        streaming=streaming,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
    )

    max_input_tokens = int(cfg_get(cfg, "data.max_input_tokens"))
    max_target_tokens = int(cfg_get(cfg, "data.max_target_tokens"))

    caps = {
        "train": cfg_get(cfg, "data.max_train_examples"),
        "validation": cfg_get(cfg, "data.max_valid_examples"),
        "test": cfg_get(cfg, "data.max_test_examples"),
    }

    # Prepare two processed dataset variants:
    # - baseline: no section tokens
    # - section_tokens: section markers inserted (used by main + ablation_no_routing)
    out_dir = processed_dir
    meta = {
        "dataset": {"name": dataset_name, "config": dataset_config, "streaming": streaming},
        "tokenizer_base": base_model_name,
        "max_input_tokens": max_input_tokens,
        "max_target_tokens": max_target_tokens,
        "roles": roles,
        "caps": caps,
    }
    write_json(out_dir / "preprocess_config.json", meta)

    def _prepare_variant(variant_name: str, use_section_tokens: bool) -> None:
        vdir = out_dir / variant_name
        if vdir.exists() and (vdir / "dataset_dict.json").exists():
            return
        vdir.mkdir(parents=True, exist_ok=True)

        dd = DatasetDict()
        for split_name, raw_split in raw.items():
            if split_name not in {"train", "validation", "test"}:
                continue
            dd[split_name] = prepare_split(
                raw_split,
                tokenizer=tokenizer,
                mapper=mapper,
                roles=roles,
                use_section_tokens=use_section_tokens,
                max_input_tokens=max_input_tokens,
                max_target_tokens=max_target_tokens,
                max_examples=caps[split_name],
            )
        dd.save_to_disk(str(vdir))

    _prepare_variant("baseline_no_section_tokens", use_section_tokens=False)
    _prepare_variant("with_section_tokens", use_section_tokens=True)

    # Save tokenizer (with added special tokens) alongside processed data for consistency.
    tok_dir = out_dir / "tokenizer"
    tokenizer.save_pretrained(str(tok_dir))

    checksums = compute_tree_checksums(out_dir)
    write_json(out_dir / "checksums.json", {"root": str(out_dir), "sha256": checksums})


if __name__ == "__main__":
    main()
