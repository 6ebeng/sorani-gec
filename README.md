# Sorani GEC — Agreement-Aware Grammatical Error Correction for Central Kurdish

> **Agreement-Aware Grammatical Error Correction for Central Kurdish (Sorani):
> A Morphology-Driven Neural Approach**
>
> Tishko Salah Hawez · MSc Software Engineering · University of Kurdistan Hewlêr · 2026
> Supervisor: Dr. Hossein Hassani

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Models on HF](https://img.shields.io/badge/🤗%20Models-Tishko%2Fsorani--gec-yellow)](https://huggingface.co/Tishko/sorani-gec)
[![Tests](https://img.shields.io/badge/tests-668%20passing-brightgreen)]()
[![Wiki](https://img.shields.io/badge/docs-wiki-blue)](https://github.com/6ebeng/sorani-gec/wiki)

The first neural grammatical error correction system for Central Kurdish (Sorani). Trained on a fully synthetic corpus built from proofread Kurdish dissertations and textbooks (26,841 training pairs in the final campaign), targeting agreement errors that arise from the split-ergative morphosyntax of Sorani.

**Full documentation lives in the [project wiki](https://github.com/6ebeng/sorani-gec/wiki)** — pipeline, architecture, training campaigns, metrics, and troubleshooting. Wiki sources are versioned in [docs/wiki/](docs/wiki/).

---

## Results

**Definitive — clean training campaign** (26,841 train pairs, fixed 647-sentence test set, span F₀.₅ at `max_length=512`, seed 42; `results/campaign_3_clean_final/`):

| Model                   | F₀.₅       | Precision  | Recall |
| ----------------------- | ---------- | ---------- | ------ |
| ByT5-small baseline     | 0.5057     | 0.6215     | 0.2898 |
| ByT5-small + morphology | **0.5105** | **0.6359** | 0.2854 |

Δ F₀.₅ = +0.0048; paired bootstrap p = 0.39 (not significant at this scale and data size).

**Prior campaign — multiseed** (3 seeds × 2 models on the 5,253-pair splits_v2; checkpoints on HF; `results/campaign_2_multiseed/`): span F₀.₅ 0.165 (baseline) vs 0.177 (morphology), p = 0.08. The clean campaign fixed three data/eval bugs (category-label contamination, 128-byte eval truncation, 256-byte target truncation) and retrained at scale — see the [Training-Campaigns wiki page](https://github.com/6ebeng/sorani-gec/wiki/Training-Campaigns) and the campaign mapping in [results/README.md](results/README.md).

**Human evaluation:** 37 native Sorani raters, 60 blind pairs; morphology-aware edits scored 2.616 vs 2.529 mean grammaticality (1–3 scale); max pairwise Cohen's κ = 0.7073.

---

## Pre-trained Models

The multiseed-campaign checkpoints (3 seeds × 2 variants, trained on splits_v2) are on Hugging Face under the legacy `phase_d/` prefix. The clean-campaign checkpoints (the 0.51 F₀.₅ models) are managed via [upload_to_hf.py](upload_to_hf.py).

```bash
pip install huggingface_hub
python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download("Tishko/sorani-gec", local_dir="./hf_models")
EOF
```

Or load directly:

```python
from transformers import AutoTokenizer, T5ForConditionalGeneration
import torch

tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")
model = T5ForConditionalGeneration.from_pretrained("google/byt5-small")

state = torch.load("hf_models/phase_d/baseline_seed42/best_model.pt", map_location="cpu")
sd = state.get("model_state_dict", state)
model.load_state_dict(sd, strict=False)
model.eval()

sentence = "کوڕەکە دەڕۆن"   # corrupted: singular subject, plural verb
inputs = tokenizer(sentence, return_tensors="pt")
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=64)
print(tokenizer.decode(out[0], skip_special_tokens=True))
# → کوڕەکە دەڕوا  (corrected)
```

---

## Installation

```bash
git clone https://github.com/6ebeng/sorani-gec.git
cd sorani-gec
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -e ".[dev]"
```

GPU training requires CUDA 11.8+. See `requirements-gpu.txt`.

---

## Project Structure

```
sorani-gec/
├── configs/
│   └── default.yaml             # Training / evaluation configuration
├── src/
│   ├── data/
│   │   ├── collector.py         # Corpus collector (dissertations + textbooks)
│   │   ├── normalizer.py        # Arabic-script Unicode normalization
│   │   ├── sanitizer.py         # 9-stage corpus sanitizer
│   │   ├── sorani_detector.py   # Sorani vs. non-Sorani language filter
│   │   ├── splitter.py          # Stratified train/dev/test splitting
│   │   └── spell_checker.py     # Hunspell-based spell checker
│   ├── errors/                  # 25 rule-based error generators
│   │   ├── base.py              # BaseErrorGenerator ABC
│   │   ├── subject_verb.py      # Subject-verb number disagreement
│   │   ├── tense_agreement.py   # Tense agreement (split-ergative)
│   │   ├── clitic.py            # Pronominal clitic form errors
│   │   ├── noun_adjective.py    # Noun-adjective ezafe mismatch
│   │   ├── word_order.py        # SOV word-order violations
│   │   ├── orthography.py       # Orthographic / script errors
│   │   ├── spelling_confusion.py# Morphophonemic character confusion
│   │   └── pipeline.py          # Synthetic corpus generation pipeline
│   ├── morphology/
│   │   ├── analyzer.py          # Rule-based morphological analyzer
│   │   ├── features.py          # 9 features per word (person, number, tense …)
│   │   ├── graph.py             # AgreementGraph (33 typed edges)
│   │   ├── builder.py           # Graph construction from analyzed sentences
│   │   └── constants.py         # Linguistic constants (F#1–F#378)
│   ├── model/
│   │   ├── baseline.py          # BaselineGEC — ByT5-small, byte-level only
│   │   ├── morphology_aware.py  # MorphologyAwareGEC — ByT5 + morph features
│   │   │                        #   + agreement graph + auxiliary loss
│   │   └── ensemble.py          # Majority-vote / best-score ensemble
│   └── evaluation/
│       ├── f05_scorer.py        # Span F₀.₅ (precision-weighted)
│       ├── agreement_accuracy.py# 14 Sorani-specific agreement checks
│       ├── gleu_scorer.py       # GLEU with bootstrap confidence intervals
│       ├── m2_scorer.py         # M² scorer
│       └── inter_rater.py       # Cohen's κ / Fleiss' κ
├── scripts/
│   ├── 01_collect_data.py       # Step 1 — collect raw Sorani text
│   ├── 01b_sanitize.py          # Step 1b — sanitize corpus
│   ├── 02_normalize.py          # Step 2 — normalize Unicode
│   ├── 03_generate_errors.py    # Step 3 — generate synthetic error pairs
│   ├── 04_split_data.py         # Step 4 — stratified splits
│   ├── 05_train_baseline.py     # Step 5 — train baseline model
│   ├── 06_train_morphaware.py   # Step 6 — train morphology-aware model
│   ├── 07_evaluate.py           # Step 7 — evaluate (F₀.₅, GLEU, M², agr.)
│   ├── 08_ablation.py           # Step 8 — ablation studies
│   ├── 10_infer.py              # CLI inference on a sentence
│   ├── 11_hash_data.py          # SHA-256 data integrity check
│   ├── create_splits_v2.py      # Canonical splits (single-edit, dedup, manifest)
│   ├── 14_build_scaled_train.py # Scaled 26,841-pair training pool
│   ├── train_campaign_3_clean_final.sh  # Clean-campaign training (definitive)
│   ├── train_campaign_2_multiseed.sh    # Multiseed campaign training driver
│   ├── eval_seed42_512.py       # Clean-campaign span scorer
│   └── analyze_human_eval.py    # 37-rater study analysis (κ, τ-b)
├── docs/wiki/                   # Wiki sources (published to GitHub wiki)
├── tests/                       # 626 tests (+42 in ../web/tests)
├── results/                     # See results/README.md for the campaign map
│   ├── campaign_3_clean_final/  # Definitive clean campaign (seed 42)
│   ├── campaign_2_multiseed/    # Prior 3-seed campaign (checkpoints on HF)
│   ├── baselines/               # Non-neural baseline results
│   ├── human_eval/              # 37-rater evaluation data
│   └── ablation/                # Ablation metrics
├── upload_to_hf.py              # Checkpoint upload to Hugging Face
├── Dockerfile
├── pyproject.toml
└── Makefile
```

> **Note:** `data/raw/`, `data/clean/`, `data/synthetic/`, and `data/splits/` are gitignored and must be generated locally by running the pipeline scripts above.

---

## Running the Pipeline

```bash
python scripts/01_collect_data.py
python scripts/01b_sanitize.py
python scripts/02_normalize.py
python scripts/03_generate_errors.py
python scripts/04_split_data.py
python scripts/05_train_baseline.py    # requires GPU
python scripts/06_train_morphaware.py  # requires GPU
python scripts/07_evaluate.py
```

All steps can also be run via:

```bash
make reproduce
```

To reproduce the published campaigns exactly (canonical splits, scaled pool, clean-campaign training), follow the [Reproducibility wiki page](https://github.com/6ebeng/sorani-gec/wiki/Reproducibility).

Key training hyperparameters are in `configs/default.yaml`. GPU training was done on an RTX 5090 (vast.ai) in FP32; FP16 produced NaN loss on this dataset.

---

## Docker

```bash
# GPU container (CUDA 11.8)
docker compose up --build

# Standalone
docker build -t sorani-gec .
docker run --gpus all -p 7860:7860 sorani-gec
```

---

## Testing

```bash
make test-all                                 # all 668 tests (626 core + 42 web)
pytest tests/ -v                              # core suite
pytest tests/test_error_generators.py -v     # error generators only
pytest tests/test_morphology.py -v           # morphological analyzer
pytest tests/test_evaluation.py -v           # metrics
```

---

## Repository Layout

| Resource           | Link                                      |
| ------------------ | ----------------------------------------- |
| Code (this repo)   | https://github.com/6ebeng/sorani-gec      |
| Documentation wiki | https://github.com/6ebeng/sorani-gec/wiki |
| Pre-trained models | https://huggingface.co/Tishko/sorani-gec  |
| Thesis document    | https://github.com/6ebeng/research_thesis |

---

## Citation

```bibtex
@mastersthesis{hawez2026soranigec,
  author  = {Tishko Salah Hawez},
  title   = {Agreement-Aware Grammatical Error Correction for Central Kurdish
             (Sorani): A Morphology-Driven Neural Approach},
  school  = {University of Kurdistan Hewl\^{e}r},
  year    = {2026},
  type    = {MSc Thesis},
}
```

---

## License

MIT License — see [LICENSE](LICENSE).
