# Getting Started

## Requirements

- Python 3.10+
- ~4 GB disk for the base install; ~40 GB if you keep training checkpoints
- GPU with CUDA for training (inference works on CPU). Training for the thesis ran on a rented RTX 5090 (vast.ai) in **FP32** — FP16 produced NaN loss with ByT5 on this data
- Windows, Linux, and macOS all work for the data pipeline and inference

## Install

```bash
git clone https://github.com/6ebeng/sorani-gec.git
cd sorani-gec
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -e ".[dev]"
```

GPU training dependencies (CUDA build of PyTorch) are pinned in `requirements-gpu.txt`. Exact versions used for the thesis runs are frozen in `requirements-lock.txt`.

### Windows console encoding

Sorani text is Arabic-script. Before running any script that prints Kurdish text on Windows PowerShell:

```powershell
$env:PYTHONIOENCODING = "utf-8"
```

All project files are UTF-8. See [[Troubleshooting]] if you see mojibake.

## Quick smoke test (no GPU, ~5–10 min)

Runs a miniature end-to-end pipeline — collect → generate → split → train a tiny model → evaluate — to prove the install works:

```bash
python scripts/smoke_test_pipeline.py
# or on Windows:
scripts\smoke_test.ps1
```

## First inference

Download a checkpoint from [Hugging Face](https://huggingface.co/Tishko/sorani-gec) (see [[Reproducibility]]), then:

```bash
python scripts/10_infer.py --model-path path/to/best_model.pt --sentence "کوڕەکە دەڕۆن"
```

## Docker

```bash
# GPU container
docker build -t sorani-gec .
docker run --gpus all -p 7860:7860 sorani-gec
```

`Implementation/docker-compose.yml` (one level up, in the parent workspace) defines services for the web demo, training, evaluation, and tests.

## Repository layout

```
sorani-gec/
├── configs/          # YAML training/eval configs (default.yaml is canonical)
├── data/             # Gitignored corpora and splits (regenerate via pipeline)
├── docs/wiki/        # Source of this wiki
├── results/          # Metrics JSONs (tracked) + checkpoints (gitignored)
├── scripts/          # Numbered pipeline steps + campaign/analysis scripts
├── src/
│   ├── data/         # Collector, sanitizer, normalizer, splitter, detector
│   ├── errors/       # 25 rule-based error generators
│   ├── evaluation/   # F0.5, GLEU, M2, agreement accuracy, inter-rater
│   ├── model/        # BaselineGEC, MorphologyAwareGEC, EnsembleGEC
│   └── morphology/   # Analyzer, features, agreement graph, lexicon
├── tests/            # 626 tests (plus 42 in ../web/tests)
└── Makefile          # make reproduce, make test, ...
```

The interactive demo lives in the sibling folder `../web/` — see [[Web-Interface]].

Next: [[Data-Pipeline]]
