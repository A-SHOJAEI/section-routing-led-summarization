# Report: Section-Routed LED Summarization

Config: `configs/smoke.yaml`

## Summary Table (Mean [95% bootstrap CI])

| Run | N | ROUGE-1 F | ROUGE-2 F | ROUGE-L F | BERTScore F1 | NLI (ent-contr) |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 24 | 0.3166 [0.2933, 0.3398] | 0.0829 [0.0613, 0.1081] | 0.1891 [0.1730, 0.2151] | 0.8701 [0.8636, 0.8758] | 0.7393 [0.5051, 0.9124] |
| main | 24 | 0.2325 [0.1928, 0.2683] | 0.0496 [0.0346, 0.0622] | 0.1505 [0.1296, 0.1679] | 0.8549 [0.8428, 0.8632] | 0.7404 [0.5729, 0.8991] |
| ablation_no_routing | 24 | 0.2447 [0.2088, 0.2805] | 0.0508 [0.0383, 0.0614] | 0.1545 [0.1386, 0.1703] | 0.8565 [0.8482, 0.8646] | 0.6991 [0.5217, 0.8793] |

## Implemented Comparisons

- Baseline: no section tokens + single shared LoRA (no routing).
- Main: section tokens + routed LoRA (per-role expert) + auxiliary role classification loss.
- Ablation (No routing): section tokens kept + single shared LoRA (routing disabled).

