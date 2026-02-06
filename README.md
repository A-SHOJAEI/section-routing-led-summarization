# Section-Routed LED Summarization (Routed LoRA + Section Markers)

## Problem
Long-document summarization of scientific papers (arXiv). Standard seq2seq summarizers treat the input as a flat token stream; this repo tests whether injecting *section structure* and *routing adaptation parameters by section role* changes summary quality and factuality proxies.

This is an end-to-end, runnable implementation (data prep -> train -> eval -> report) with three explicitly-defined runs and measured results in `artifacts/results.json` and `artifacts/report.md`.

## Dataset (Provenance + License Caveat)
- Source: Hugging Face Datasets `armanc/scientific_papers`, config `arxiv` (loaded in `src/srls/data/prepare.py`).
- Fields used:
  - Preferred: `section_names` + `sections` (paired lists) and `abstract` as the target summary.
  - Fallback: `article` (single field) if structured sections are not present (still paired with `abstract`).
- License caveat: the dataset is arXiv-derived and document-level redistribution rights are not uniform. This repo’s code is MIT-licensed (`LICENSE`), but dataset/model licensing is upstream-dependent; treat dataset redistribution and derived artifacts cautiously.

## Method (As Implemented)
### Section Tokens + Per-Token Roles
Implemented in `src/srls/data/prepare.py` and `src/srls/data/roles.py`.
- A regex-based mapper buckets raw section headings into roles:
  - `intro`, `methods`, `results`, `discussion`, `conclusion`, `other`.
- When enabled, preprocessing builds the encoder source text by prefixing each section with a special marker token:
  - `<sec:{role}>` followed by the section text.
- Tokenization adds the role markers as `additional_special_tokens` to the LED tokenizer.
- `role_ids` are derived by scanning the tokenized `input_ids` left-to-right:
  - each time a section marker token is encountered, the “current role” is updated;
  - every subsequent token gets that role id until the next marker.

### LED Global Attention Mask
LED uses a `global_attention_mask`. This repo sets:
- global attention on position `0` (the first token) and
- global attention on any `<sec:...>` marker tokens.

### LoRA (Custom Implementation)
Implemented in `src/srls/models/routed_lora.py`.
- Base model: `allenai/led-base-16384` (`configs/*.yaml`).
- LoRA is applied by *wrapping* encoder `nn.Linear` projection modules with a `RoutedLoRALinear`:
  - LoRA weights are `A` (down) and `B` (up), initialized as:
    - `A ~ Normal(0, 0.02)`, `B = 0`.
  - Scaling is `alpha / rank`.
- Only LoRA parameters and the auxiliary head are trainable (all base model weights are frozen in `src/srls/train.py`).

### Routed Experts (Per-Role LoRA on the Encoder)
Implemented in `src/srls/models/build.py` and `src/srls/models/routing_state.py`.
- Routing is *token-level* using the precomputed `role_ids`.
- “Experts” are separate LoRA parameter sets per role:
  - `num_experts = len(section_roles)` when `routed=True`
  - `num_experts = 1` when `routed=False` (shared LoRA).
- Routing is applied only to **encoder** linears whose leaf module names match the configured targets.
  - Config uses `target_modules: ["q_proj", "v_proj"]`, but LED’s Longformer encoder names these `query` / `value`. The builder maps `q_proj -> query` and `v_proj -> value`, and also targets the `_global` variants.
- Decoder is not LoRA-routed in this implementation.

### Auxiliary Loss (Role Classification at Section Markers)
Implemented in `src/srls/models/led_wrapper.py`.
- When `aux_loss_weight > 0`, an MLP head `role_head: Linear(d_model -> num_roles)` is trained.
- The auxiliary loss is computed only at positions where `input_ids` equals any `<sec:...>` token id:
  - `cross_entropy(role_head(encoder_hidden_at_marker), role_id_at_marker)`
- This term is added to the model loss as: `loss += aux_loss_weight * aux_loss`.

## Runs / Baselines / Ablations (Exact Definitions)
All three runs are wired in `src/srls/train.py#get_run_spec` and trained/evaluated via the Makefile.

- `baseline` (`outputs/baseline_shared_lora_no_section_tokens/`)
  - Data variant: `data/processed/*/baseline_no_section_tokens/` (no `<sec:...>` markers)
  - LoRA: single shared LoRA (`routed=False`)
  - Aux loss: **disabled** (`aux_loss_weight=0.0` forced in code)

- `main` (`outputs/main_routed_lora_section_tokens_aux_loss/`)
  - Data variant: `data/processed/*/with_section_tokens/` (markers enabled)
  - LoRA: routed per-role experts (`routed=True`)
  - Aux loss: enabled (from config, `model.aux_loss_weight`)

- `ablation_no_routing` (`outputs/ablation_no_routing_section_tokens_shared_lora/`)
  - Data variant: `data/processed/*/with_section_tokens/` (markers enabled)
  - LoRA: single shared LoRA (`routed=False`)
  - Aux loss: enabled (from config, `model.aux_loss_weight`)
  - Note: this ablation isolates “routing on/off” while keeping section markers and the auxiliary role loss *on*.

## Evaluation (Metrics + CI)
Implemented in `src/srls/eval.py`.
- ROUGE-1/2/L: `rouge_score` f-measure per example (with stemming), mean + 95% bootstrap CI.
- BERTScore F1: `bert-score` per example, mean + 95% bootstrap CI.
- NLI factuality proxy: `entailment_probability - contradiction_probability` from an MNLI classifier, mean + 95% bootstrap CI.
  - Premise: source text truncated to `eval.max_source_chars_for_nli`
  - Hypothesis: generated summary
- Bootstrap: `eval.bootstrap_samples` resamples with replacement (seed `project.seed`).

## Results (Measured)
The following table is copied verbatim from `artifacts/report.md` (generated from `configs/smoke.yaml`).

| Run | N | ROUGE-1 F | ROUGE-2 F | ROUGE-L F | BERTScore F1 | NLI (ent-contr) |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 24 | 0.3166 [0.2933, 0.3398] | 0.0829 [0.0613, 0.1081] | 0.1891 [0.1730, 0.2151] | 0.8701 [0.8636, 0.8758] | 0.7393 [0.5051, 0.9124] |
| main | 24 | 0.2325 [0.1928, 0.2683] | 0.0496 [0.0346, 0.0622] | 0.1505 [0.1296, 0.1679] | 0.8549 [0.8428, 0.8632] | 0.7404 [0.5729, 0.8991] |
| ablation_no_routing | 24 | 0.2447 [0.2088, 0.2805] | 0.0508 [0.0383, 0.0614] | 0.1545 [0.1386, 0.1703] | 0.8565 [0.8482, 0.8646] | 0.6991 [0.5217, 0.8793] |

## Reproducibility (Makefile + CONFIG)
All entrypoints are exposed as Makefile targets (`Makefile`). `CONFIG` selects the YAML config (defaults to `configs/smoke.yaml`).

```bash
# End-to-end: venv + deps + data + train + eval + report
make all

# Run the larger config (non-streaming, larger token lengths/steps)
make all CONFIG=configs/experiment.yaml
```

Individual steps:
```bash
make setup                 # creates .venv, installs pinned pip + torch (via scripts/install_torch.sh), installs deps + package
make data  CONFIG=...       # prepares data/processed/*/{baseline_no_section_tokens,with_section_tokens} and tokenizer; writes checksums
make train CONFIG=...       # trains baseline, main, ablation_no_routing
make eval  CONFIG=...       # writes outputs/<run>/eval/results.json and predictions.jsonl
make report CONFIG=...      # writes report.md (summary table across runs)
make clean                 # removes .venv, data/processed, outputs, report.md
```

Config knobs that materially change behavior (see `configs/*.yaml`):
- `data.streaming` and `data.max_*_examples`: streaming requires caps per split.
- `data.max_input_tokens`, `data.max_target_tokens`: truncation limits during preprocessing.
- `model.section_roles`: defines the role vocabulary and number of routed experts.
- `model.target_modules`: which encoder projections get LoRA-wrapped (mapped to LED’s `query`/`value` + `_global` variants).
- `model.aux_loss_weight`: enables/disables auxiliary role classification loss.
- `train.*`: optimizer schedule + step budget (smoke uses `max_steps: 6`).
- `eval.*`: decoding and metrics models (BERTScore/NLI) + bootstrap samples.

## Hardware / Software Environment (From `artifacts/results.json`)
Measured on:
- OS: `Linux-6.17.0-14-generic-x86_64-with-glibc2.39`
- Python: `3.12.3`
- PyTorch: `2.10.0+cu126`
- CUDA: available (`true`), devices: `2`, GPU[0]: `NVIDIA GeForce RTX 3090`
- Decoding settings (smoke): `max_new_tokens=128`, `num_beams=2`

## Repo Outputs (What Gets Written Where)
- Processed datasets and tokenizer:
  - `data/processed/<name>/baseline_no_section_tokens/`
  - `data/processed/<name>/with_section_tokens/`
  - `data/processed/<name>/tokenizer/`
  - `data/processed/<name>/checksums.json` (validated by `make data`)
- Training:
  - `outputs/<run>/model/model.pt` (checkpoint)
  - `outputs/<run>/train_metrics.json`
  - `outputs/<run>/valid_metrics.json`
- Evaluation:
  - `outputs/<run>/eval/results.json`
  - `outputs/<run>/eval/predictions.jsonl` (for error analysis)
- Aggregated report:
  - `report.md` (generated by `make report`)

## Limitations (Observed / Structural)
- The only measured results in this repo are from `configs/smoke.yaml` (N=24 test examples, `train.max_steps=6`). These numbers are useful for verifying wiring and regression-testing, not for claiming a modeling improvement.
- In the measured smoke results, the routed+aux model (`main`) underperforms the baseline on ROUGE and BERTScore; this is consistent with “not trained enough” and/or mis-specified routing for this setup.
- `ablation_no_routing` still uses the auxiliary role loss (`aux_loss_weight=0.1` in the smoke config), so it is not a pure “section tokens only” baseline.
- Routing is only applied to encoder attention projections (`query`/`value` and `_global` variants). Decoder adaptations and non-attention modules are untouched.
- Section role mapping is heuristic regex over headings, not a learned segmenter and not aligned to any arXiv LaTeX structure beyond what the dataset provides.
- The NLI metric is a proxy using an MNLI classifier (`typeform/distilbert-base-uncased-mnli`) with the source truncated by character count; it is not a calibrated factuality evaluation.

## Next Experiments (Concrete, Repo-Local)
1. Add two new runs in `src/srls/train.py#get_run_spec` to isolate factors cleanly:
   - `ablation_no_aux`: section tokens on, routing off, `aux_loss_weight=0.0`
   - `ablation_routed_no_aux`: section tokens on, routing on, `aux_loss_weight=0.0`
2. Run the full-length config (`configs/experiment.yaml`) and report the same table (increase `eval.bootstrap_samples` to reduce CI noise).
3. Expand LoRA targets in `configs/experiment.yaml` (and/or in code) to include `key` and `output` projections and measure whether routing helps beyond `query`/`value`.
4. Add expert-utilization logging (counts of tokens per role per batch, and per-expert token totals) to verify routing is actually exercised in practice.
5. Slice evaluation by document structure:
   - examples with true `section_names/sections` vs fallback `article`
   - role marker frequency and per-role attention distribution (requires small instrumentation).
6. Qualitative error analysis:
   - diff `outputs/*/eval/predictions.jsonl` for hallucinations and section-specific omissions, then adjust `aux_loss_weight`, `lora_rank`, and marker placement.

Last updated (UTC): 2026-02-06 19:18:06
