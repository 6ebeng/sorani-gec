# Sorani GEC Wiki

**Agreement-Aware Grammatical Error Correction for Central Kurdish (Sorani): A Morphology-Driven Neural Approach**

MSc thesis project by Tishko Salah Hawez, University of Kurdistan Hewlêr (UKH), 2026. Supervised by Dr. Hossein Hassani. This is the first neural grammatical error correction (GEC) system for Central Kurdish.

## What this project does

Sorani Kurdish has a split-ergative agreement system: person/number marking moves between verb affixes and pronominal clitics depending on tense and transitivity. Standard byte-level seq2seq models see none of that structure. This project:

1. Builds a synthetic GEC corpus from proofread Kurdish dissertations and textbooks (OCR → sanitize → normalize → inject errors with 25 rule-based generators).
2. Trains two ByT5-small variants: a plain byte-level **baseline** and a **morphology-aware** model that adds 9 morphological features per word, a 33-edge agreement graph, and an auxiliary agreement-prediction loss.
3. Evaluates with span F₀.₅, GLEU, M², a 14-check Sorani agreement accuracy metric, paired bootstrap significance tests, non-neural baselines, an LLM baseline (Aya-Expanse-8B), and a 37-rater human study.

## Headline result (clean campaign, June 2026)

| Model | Span F₀.₅ | P | R | TP | FP | FN |
|---|---|---|---|---|---|---|
| ByT5-small baseline | 0.5057 | 0.6215 | 0.2898 | 133 | 81 | 326 |
| ByT5-small + morphology | **0.5105** | **0.6359** | 0.2854 | 131 | 75 | 328 |

Trained on 26,841 pairs (`splits_scaled`), evaluated on the fixed 647-sentence test set at `max_length=512`, seed 42. Δ F₀.₅ = +0.0048, paired bootstrap p = 0.39 (not significant). Human raters scored morphology-aware edits 2.616 vs 2.529 for the baseline on a 1–3 grammaticality scale (37 raters, 60 blind pairs, max pairwise κ = 0.7073).

See [[Results]] for all campaigns, ablations, and baselines.

## Pages

| Page | Contents |
|---|---|
| [[Getting-Started]] | Install, environment, smoke test, Docker |
| [[Data-Pipeline]] | Corpus sources, sanitization, normalization, splits chronology |
| [[Error-Generation]] | The 25 rule-based error generators and the synthesis pipeline |
| [[Morphological-Analysis]] | Analyzer, 9 features, agreement graph, lexicon |
| [[Model-Architecture]] | ByT5 baseline, morphology-aware variant, ensemble |
| [[Training-Campaigns]] | All training campaigns in order, with exact commands |
| [[Evaluation-Metrics]] | Span F₀.₅, GLEU, M², agreement accuracy, bootstrap |
| [[Results]] | Definitive numbers, prior campaigns, ablations, human eval |
| [[Web-Interface]] | Gradio demo, REST API, annotation server |
| [[Testing]] | The 668-test suite and how to run it |
| [[Reproducibility]] | One-command reproduction, data hashes, HF checkpoints |
| [[Troubleshooting]] | FP16 NaN, Windows encoding, Hunspell, CUDA issues |

## Links

- Code: <https://github.com/6ebeng/sorani-gec>
- Pre-trained models: <https://huggingface.co/Tishko/sorani-gec>
- License: MIT
