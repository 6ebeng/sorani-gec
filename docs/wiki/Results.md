# Results

Every number here is regenerable from tracked JSONs in `results/`. The definitive campaign is the **clean campaign** (`results/phase2_clean/`); everything else is context.

## Definitive: clean campaign (June 2026)

Test: full 647-pair splits_v2-compatible test set, span F₀.₅, `max_length=512`, seed 42. Source: `results/phase2_clean/eval_summary_512.json`.

| Model | F₀.₅ | P | R | TP | FP | FN |
|---|---|---|---|---|---|---|
| ByT5-small baseline | 0.5057 | 0.6215 | 0.2898 | 133 | 81 | 326 |
| ByT5-small + morphology | **0.5105** | **0.6359** | 0.2854 | 131 | 75 | 328 |

Δ F₀.₅ = +0.0048 in favour of morphology; paired bootstrap (10,000 resamples) p = 0.39 — the margin does not clear significance. The honest headline: after all data bugs are fixed, morphology injection no longer *hurts* (it did in Phase D), and trends slightly ahead on precision.

## Prior campaign: Phase D (3 seeds, splits_v2, 5,253 train pairs)

Source: `results/phase_d/eval_summary.json` (word-level), `results/phase3_metrics.json` (span recompute).

| Model | Span F₀.₅ (mean ± std, 3 seeds) | Word F₀.₅ |
|---|---|---|
| Baseline | 0.1651 ± 0.0077 | ≈0.080 |
| Morphology-aware | 0.1767 ± 0.0094 | ≈0.067–0.088 |

Bootstrap p = 0.08 (full set), p = 0.16 (edited subset). At this data scale both models are starved; the thesis Chapter 7 dissects why (5,253 pairs cannot feed a 300M-parameter model).

## Non-neural and LLM baselines

| System | Span F₀.₅ |
|---|---|
| Copy | 0.000 |
| Hunspell | 0.009 |
| Reverse-rule | 0.085 |
| N-gram LM | 0.139 |
| Aya-Expanse-8B (zero-shot) | below neural models; see thesis Ch. 6 |

Source: `results/baselines/baseline_summary.json`, `bootstrap_pvalues.json`.

## Human evaluation (37 raters)

Blind study: 60 pairs (20 baseline-edited, 20 morphaware-edited, 20 both-agree), each shown as source + one correction, system identity hidden. Ratings on a 3-point scale: دروست (correct) / بەشێکی دروست (partial) / هەڵە (wrong). Source: `results/human_eval/metric_validation.json`.

| System | Mean grammaticality (1–3) |
|---|---|
| Baseline | 2.529 |
| Morphology-aware | **2.616** |
| Both-agree pairs | 2.613 |

- Max pairwise Cohen's κ = 0.7073 (substantial agreement; meets the pre-registered criterion).
- Metric validation (R15): Kendall τ-b between the 14-check agreement metric and human ordinal judgement = 0.0 — the automated agreement metric does **not** track human grammaticality on this sample. Reported as a negative finding.
- Automated proxy raters were excluded (`includes_automated_proxy: false`).

## Ablations (`results/ablation/ablation_summary.json`)

Run on the early (pre-cleanup) data, so absolute values are tiny; the *relative* ordering informed the final configuration:

- **λ (agreement-loss weight) sweep** → λ=0.0 best on dev; the auxiliary loss as formulated does not help.
- **Individual features** (person/number/tense probes) and **data-size variation** (10K–50K) — data size dominates every architectural choice.
- **Curriculum learning** — no reliable gain (`src/data/curriculum.py`).

## Supporting analyses

| Artifact | Produced by |
|---|---|
| `results/phase_d/agreement_subset_rescore.json` | `13_agreement_subset_rescore.py` — agreement-error-only rescore |
| `results/phase_d/edited_subset_recomputed.json` | `recompute_edited_subset_phase_d.py` — 397-pair edited subset |
| `results/oov_rate.json` | `measure_oov_rate.py` — OOV rate vs lexicon |
| `results/ocr_audit/` | `ocr_audit.py` — CER/WER of the OCR sources per university |
| `results/data_diagnosis/` | `diagnose_data.py` — leakage/trivial-pair audit |
| `results/baselines/natural_eval.json` | `eval_baselines_natural.py` — natural-sentence test set |

Next: [[Web-Interface]]
