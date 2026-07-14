# Reproducibility

## One command

```bash
make reproduce
```

Chains collect → sanitize → normalize → generate → split → stats → train-baseline → train-morphaware → evaluate → ablation. Training steps need a GPU; everything else is CPU.

## Reproducing the published campaigns exactly

The generic pipeline reproduces the *approach*; the published numbers came from specific campaign scripts (see [[Training-Campaigns]] for context):

```bash
# 1. Canonical splits (dev/test are frozen; SHA-256 manifest written)
python scripts/create_splits_v2.py
python scripts/augment_test_v2.py

# 2. Scaled training pool for the clean campaign
python scripts/15_clean_corpus.py
python scripts/14_build_scaled_train.py     # → data/splits_scaled (26,841 train)

# 3. Definitive training run (RTX 5090-class GPU, FP32)
bash scripts/phase2_retrain.sh --skip-build --batch 8 --accum 16 \
     --max-len 512 --results-dir results/phase2_clean --seeds 42

# 4. Definitive evaluation
python scripts/eval_seed42_512.py           # → eval_summary_512.json
```

Seeds: 42 (clean campaign); 42/123/777 (Phase D). Every random path seeds `random`, `numpy`, and `torch`.

## Data integrity

```bash
python scripts/11_hash_data.py    # SHA-256 manifest of all data artifacts
python scripts/diagnose_data.py   # leakage / trivial-pair / distribution audit
```

`create_splits_v2.py` embeds hashes in the split manifest — dev/test are byte-identical across splits_v2 and splits_scaled, so all campaigns compare on the same test set.

## Pre-trained checkpoints

Phase D checkpoints (3 seeds × 2 variants) are on Hugging Face:

```python
from huggingface_hub import snapshot_download
snapshot_download("Tishko/sorani-gec", local_dir="./hf_models")
```

Checkpoints are raw `torch.save` state dicts:

```python
from transformers import AutoTokenizer, T5ForConditionalGeneration
import torch

tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")
model = T5ForConditionalGeneration.from_pretrained("google/byt5-small")
state = torch.load("hf_models/phase_d/baseline_seed42/best_model.pt", map_location="cpu")
model.load_state_dict(state.get("model_state_dict", state), strict=False)
model.eval()
```

Note the campaign distinction: the HF checkpoints are **Phase D** (span F₀.₅ ≈ 0.17). The clean-campaign checkpoints (F₀.₅ ≈ 0.51) live in `results/phase2_clean/*/best_model.pt`; upload is managed by `upload_to_hf.py`.

## Environment freeze

- `requirements-lock.txt` — exact package versions from the thesis runs
- `requirements-gpu.txt` — CUDA PyTorch pins
- `Dockerfile` — CUDA 12.4 runtime image
- `.env.example` — expected environment variables

## What is intentionally not in git

`data/` and model checkpoints are gitignored (size + licensing of source texts). Tracked instead: every metrics JSON in `results/`, all human-eval ratings, hypotheses JSONLs for significance testing, and the scripts that regenerate everything else.

Next: [[Troubleshooting]]
