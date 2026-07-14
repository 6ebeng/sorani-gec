# Results Directory Guide

Training campaigns are numbered chronologically. Earlier project documents and the thesis PDF use historical names; the mapping is:

| Directory (current)         | Historical name | Campaign                                                                                                             | Data (train)           | Seeds      | Span F₀.₅ (b / m)            | Status                           |
| --------------------------- | --------------- | -------------------------------------------------------------------------------------------------------------------- | ---------------------- | ---------- | ---------------------------- | -------------------------------- |
| `metrics_remote/`           | —               | Early remote eval (April 2026)                                                                                       | splits v1              | 42         | —                            | Historical                       |
| `campaign_1_audit_retrain/` | `phase1/`       | Audit-fix retrain (warmup scaling, val-F₀.₅ selection, λ=0)                                                          | splits_v2 (5,253)      | 42/123/777 | ≈0.157 / ≈0.169              | Historical                       |
| `campaign_2_multiseed/`     | `phase_d/`      | Multiseed campaign; checkpoints on [HF](https://huggingface.co/Tishko/sorani-gec) under the legacy `phase_d/` prefix | splits_v2 (5,253)      | 42/123/777 | 0.165 / 0.177 (p=0.08)       | Prior campaign (thesis Ch. 7)    |
| `campaign_3_clean_final/`   | `phase2_clean/` | **Clean final campaign** — contamination + truncation bugs fixed, scaled data                                        | splits_scaled (26,841) | 42         | **0.5057 / 0.5105** (p=0.39) | **Definitive (thesis headline)** |

Note: "Phase D" was internal reviewer-report jargon, and the "phase 1/2/3" numbering came from a separate retrain plan; neither reflects chronology (the `phase2_clean` run happened _after_ `phase_d`). Hence the rename.

## Renamed artifacts

| Current                       | Historical                                                                                                                   |
| ----------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `campaigns_span_metrics.json` | `phase3_metrics.json` — span-aware recompute of campaigns 1–2 (JSON keys inside keep the legacy labels `phase1` / `phase_d`) |
| `splits_v2_audit.json`        | `phase2_data_audit.json` — splits_v2 integrity audit (rows 2.1–2.5)                                                          |

The thesis (Ch. 6) cites the definitive run as `scripts/phase2_retrain.sh --results-dir results/phase2_clean`; that script is now `scripts/train_campaign_3_clean_final.sh` and the directory is `campaign_3_clean_final/`. Git history preserves the old names.

## Supporting directories

| Directory                                            | Contents                                                                            |
| ---------------------------------------------------- | ----------------------------------------------------------------------------------- |
| `baselines/`                                         | Non-neural baselines (copy, hunspell, reverse-rule, n-gram LM) + bootstrap p-values |
| `human_eval/`                                        | 37-rater blind study: ratings, manifest, κ/τ analysis                               |
| `ablation/`, `ablation_partial/`                     | Consolidated ablation summary + retained per-run metrics                           |
| `data_diagnosis/`                                    | Data leakage / trivial-pair audit                                                   |
| `ocr_audit/`                                         | OCR quality (CER/WER) of dissertation sources                                       |
| `figures/`                                           | Corpus statistics plots                                                             |
| `models/`, `metrics/`                                | Default output dirs for fresh training/eval runs                                    |

Checkpoints (`*.pt`) and raw logs (`*.log`) are gitignored; metrics JSONs are tracked.
