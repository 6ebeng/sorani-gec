# Evaluation Metrics

All scorers live in `src/evaluation/`; the main entry point is `scripts/07_evaluate.py` (and campaign-specific eval scripts, see [[Training-Campaigns]]).

## Span F₀.₅ (`f05_scorer.py`) — headline metric

Edit-level precision-weighted F-score, the GEC standard (precision matters twice as much as recall):

$$F_{0.5} = \frac{1.25 \cdot P \cdot R}{0.25 \cdot P + R}$$

Two variants exist and the distinction matters when reading older results:

- **Word-level F₀.₅** — position-agnostic bag-of-edits match.
- **Span F₀.₅** — position-aware: an edit counts only if the span offsets line up. Stricter and the one reported in the thesis. Implemented in `evaluate_corpus_span`; the clean campaign scores with `scripts/eval_seed42_512.py`, and `scripts/recompute_span_metrics.py` back-fills span scores for earlier campaigns into `results/campaigns_span_metrics.json`.

## Agreement accuracy (`agreement_accuracy.py`)

Fourteen `_check_*` methods verify Sorani-specific agreement phenomena on model output (subject–verb number, clitic person/number, ezafe chains, quantifier agreement, negative concord, …). Reported two ways: legacy (all sentences) and applicable-only (sentences where a check actually fires). This metric was validated against human judgements — see [[Results]] (R15: the correlation turned out near zero, an honest negative finding).

## GLEU (`gleu_scorer.py`)

Sentence-level GLEU with bootstrap confidence intervals; secondary fluency signal.

## M² (`m2_scorer.py`)

MaxMatch scorer over M²-format gold edits (`build_m2_from_jsonl.py` produces gold files from the JSONL splits).

## Inter-rater agreement (`inter_rater.py`)

Cohen's κ (pairwise) and Fleiss' κ (multi-rater) for the human study; also percentage agreement and Kendall's τ-b in `scripts/analyze_human_eval.py`.

## Significance testing (`scripts/run_bootstrap.py`)

Paired bootstrap (10,000 resamples) over per-sentence hypotheses (`hypotheses.jsonl` dumped per run). Reports p-value and CI for the F₀.₅ delta between two systems.

## Reference systems

| System                           | Script                        | Span F₀.₅ (splits_v2 test) |
| -------------------------------- | ----------------------------- | -------------------------- |
| Copy (do nothing)                | `eval_baselines.py`           | 0.000                      |
| Hunspell spell-correct           | `eval_baselines.py`           | 0.009                      |
| Reverse-rule (invert generators) | `eval_baselines.py`           | 0.085                      |
| KenLM-style n-gram reranker      | `eval_baselines.py`           | 0.139                      |
| Aya-Expanse-8B zero-shot         | `11_evaluate_llm_baseline.py` | see thesis Ch. 6           |

Both neural models clear every non-neural baseline by a wide margin in the clean campaign (0.51 vs ≤0.14).

Next: [[Results]]
