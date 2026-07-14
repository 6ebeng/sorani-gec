# Training Campaigns

Training happened in several campaigns as data and evaluation bugs were found and fixed. Read this page to know which numbers are current and which are historical. All remote runs used a rented RTX 5090 (vast.ai), FP32.

Campaign directories are numbered chronologically. Historical project docs (and the thesis PDF) use older internal names — the mapping is in [results/README.md](https://github.com/6ebeng/sorani-gec/blob/main/results/README.md).

## Chronology

| #   | Campaign                                                                          | Data (train)               | Seeds      | Results dir (historical name)                          | Span F₀.₅ (b / m)            | Status                                     |
| --- | --------------------------------------------------------------------------------- | -------------------------- | ---------- | ------------------------------------------------------ | ---------------------------- | ------------------------------------------ |
| 1   | Early remote runs                                                                 | splits v1                  | 42         | `results/metrics_remote/`                              | —                            | Historical                                 |
| 2   | Campaign 1 — audit retrain (warmup scaling, val-F₀.₅ selection, λ=0)              | splits_v2 (5,253)          | 42/123/777 | `results/campaign_1_audit_retrain/` (`phase1`)         | ≈0.157 / ≈0.169              | Historical                                 |
| 3   | Campaign 2 — multiseed (checkpoints on HF)                                        | splits_v2 (5,253)          | 42/123/777 | `results/campaign_2_multiseed/`                        | 0.165 / 0.177 (p=0.08)       | Prior campaign — dissected in thesis Ch. 7 |
| 4   | **Campaign 3 — clean final** (contamination + truncation bugs fixed, scaled data) | **splits_scaled (26,841)** | **42**     | **`results/campaign_3_clean_final/`** (`phase2_clean`) | **0.5057 / 0.5105** (p=0.39) | **Definitive**                             |

Word-level F₀.₅ for campaign 2 sits near 0.08 (`results/campaign_2_multiseed/eval_summary.json`); the span-aware recompute is in `results/campaigns_span_metrics.json`.

## The three bugs the clean campaign fixed

1. **Category-label contamination** — domain tags (`linguistics\t`…) leaked into training targets; FP counts ≈830 per model. Fixed by `15_clean_corpus.py`.
2. **Baseline eval truncation at 128 bytes** — clipped generations scored as false positives.
3. **Target tail-truncation at 256 bytes** — training targets lost their tails at the old max_length.

After the fixes, FP dropped from ≈830 to 75–81 and F₀.₅ rose from ≈0.08 to ≈0.51 for both models.

## Exact commands

### Campaign 2 — multiseed (3 seeds × 2 models, splits_v2)

```bash
bash scripts/train_campaign_2_multiseed.sh   # trains all 6 runs → results/campaign_2_multiseed/
python scripts/eval_campaign_checkpoints.py  # per-seed eval_test.json
python scripts/dump_hypotheses.py    # hypotheses.jsonl for bootstrap
python scripts/run_bootstrap.py      # paired significance tests
```

### Campaign 3 — clean final (definitive)

```bash
python scripts/14_build_scaled_train.py         # build splits_scaled (26,841 pairs)
bash scripts/train_campaign_3_clean_final.sh --skip-build \
     --batch 8 --accum 16 --max-len 512 \
     --results-dir results/campaign_3_clean_final --seeds 42
python scripts/eval_seed42_512.py               # span scorer at max_length=512
```

(The thesis cites this run by the script's original name, `scripts/phase2_retrain.sh`, with `--results-dir results/phase2_clean`.)

GPU memory settles at ~20.7 GiB with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

### Single-model training (generic)

```bash
python scripts/05_train_baseline.py   --config configs/default.yaml
python scripts/06_train_morphaware.py --config configs/default.yaml
```

Useful flags on `06_train_morphaware.py`: `--agreement-loss-weight`, `--morph-gate-init`, `--morph-residual-init`, `--curriculum`, `--data-dir`, `--seed`.

### Hyperparameter search / ablations

```bash
make hpsearch     # scripts/12_hyperparam_search.py
make ablation     # scripts/08_ablation.py → results/ablation/
```

Remote-instance helpers (`scripts/remote_setup.sh`, `scripts/setup_remote.sh`) document the vast.ai environment setup for campaign 2 and campaign 3 respectively.

Next: [[Evaluation-Metrics]]
