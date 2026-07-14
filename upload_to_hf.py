# -*- coding: utf-8 -*-
"""
Upload Sorani GEC model checkpoints and results to Hugging Face Hub.

Uploads the campaign-2 (multiseed, formerly "Phase D") best_model.pt files
(3 seeds x 2 models) plus eval_summary.json and a model card README.

NOTE ON NAMING: the local directory is results/campaign_2_multiseed/ but the
remote HF layout keeps the historical phase_d/ prefix because those paths are
already published (download scripts in the wild reference them).

Usage:
    1. huggingface-cli login          # enter your write token once
    2. python upload_to_hf.py         # uploads everything

Repo: https://huggingface.co/Tishko/sorani-gec
"""
import os
import json
from pathlib import Path

# Force the old LFS upload path — the Xet session token expires mid-transfer
# on large files (>1 GB) and causes repeated 401 errors.
os.environ["HF_HUB_DISABLE_XET"] = "1"

from huggingface_hub import HfApi, create_repo

REPO_ID  = "Tishko/sorani-gec"
REPO_TYPE = "model"
HERE     = Path(__file__).parent
RESULTS  = HERE / "results"

# ---- Files to upload -------------------------------------------------------
# Campaign 2 (multiseed) — the 3-seed campaign on splits_v2. Local dir:
# results/campaign_2_multiseed/. Remote layout keeps the legacy phase_d/ prefix.
LOCAL_CAMPAIGN_DIR = "campaign_2_multiseed"
REMOTE_PREFIX = "phase_d"  # published HF layout; do not change
PHASE_D_SEEDS = [
    ("baseline",   "seed42"),
    ("baseline",   "seed123"),
    ("baseline",   "seed777"),
    ("morphaware", "seed42"),
    ("morphaware", "seed123"),
    ("morphaware", "seed777"),
]

# ---- Model card ------------------------------------------------------------
README = """\
---
language:
- ckb
license: mit
tags:
- grammatical-error-correction
- central-kurdish
- sorani
- byt5
- morphology
- low-resource-nlp
base_model: google/byt5-small
---

# Sorani GEC — Agreement-Aware Grammatical Error Correction for Central Kurdish

This repository contains the trained model checkpoints for the MSc thesis:

> **Agreement-Aware Grammatical Error Correction for Central Kurdish (Sorani):  
> A Morphology-Driven Neural Approach**  
> Tishko Salah Hawez · University of Kurdistan Hewlêr · 2026  
> Supervisor: Dr. Hossein Hassani

Code and evaluation results: [github.com/6ebeng/sorani-gec](https://github.com/6ebeng/sorani-gec)

---

## Models

Two variants, both fine-tuned from [google/byt5-small](https://huggingface.co/google/byt5-small)
on a synthetic Central Kurdish (Sorani) GEC corpus (~27 000 training pairs).

| Variant | Description |
|---------|-------------|
| `baseline` | Byte-level seq2seq; no linguistic features |
| `morphaware` | Same backbone + 9 morphological features per word, a 33-edge agreement graph, and an auxiliary agreement-prediction loss |

Each variant ships with three seeds (42, 123, 777) from the multiseed training
campaign on the 5,253-pair splits_v2 training set (span F₀.₅ ≈ 0.165–0.177).

The thesis headline numbers (span F₀.₅ 0.5057 baseline / 0.5105 morphology-aware)
come from a later clean campaign trained on 26,841 pairs; those checkpoints are
not yet published here.

## Results (multiseed campaign checkpoints in this repo, span F₀.₅)

| Model | F₀.₅ | Precision | Recall |
|-------|------|-----------|--------|
| Baseline (3-seed mean) | 0.165 | — | — |
| Morphology-aware (3-seed mean) | 0.177 | — | — |

Paired bootstrap p = 0.08 (not significant at α = 0.05).

Full results, ablations, human evaluation (37 native raters), and discussion
are in the thesis and in `phase_d/eval_summary.json` (word-level metrics).

## File layout

```
phase_d/
  baseline_seed42/best_model.pt
  baseline_seed123/best_model.pt
  baseline_seed777/best_model.pt
  morphaware_seed42/best_model.pt
  morphaware_seed123/best_model.pt
  morphaware_seed777/best_model.pt
  eval_summary.json
```

## Quick usage

```python
from transformers import AutoTokenizer, T5ForConditionalGeneration
import torch

model_path = "Tishko/sorani-gec"  # or a local path to best_model.pt

# The checkpoints are raw PyTorch state-dicts saved with torch.save().
# Load with the ByT5-small tokenizer:
tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")

model = T5ForConditionalGeneration.from_pretrained("google/byt5-small")
state = torch.load("phase_d/baseline_seed42/best_model.pt", map_location="cpu")
# State dict may be nested under a key — unwrap if needed:
sd = state.get("model_state_dict", state)
model.load_state_dict(sd, strict=False)
model.eval()

sentence = "کوڕەکە دەڕۆن"   # corrupted: singular subject, plural verb
inputs = tokenizer(sentence, return_tensors="pt")
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=64)
print(tokenizer.decode(out[0], skip_special_tokens=True))
# → کوڕەکە دەڕوا  (corrected: singular verb)
```

## Citation

```bibtex
@mastersthesis{hawez2026soranigec,
  author  = {Tishko Salah Hawez},
  title   = {Agreement-Aware Grammatical Error Correction for Central Kurdish
             (Sorani): A Morphology-Driven Neural Approach},
  school  = {University of Kurdistan Hewl\\^{e}r},
  year    = {2026},
  type    = {MSc Thesis},
}
```
"""

# ---- Upload logic ----------------------------------------------------------

def main():
    api = HfApi()

    # 1. Create repo (no-op if it already exists)
    print(f"Creating / verifying repo: {REPO_ID}")
    create_repo(repo_id=REPO_ID, repo_type=REPO_TYPE, exist_ok=True, private=False)

    # Build set of files already on the hub so we can skip on retry
    try:
        existing = {f.rfilename for f in api.list_repo_files(repo_id=REPO_ID, repo_type=REPO_TYPE)}
    except Exception:
        existing = set()

    # 2. Upload model card
    if "README.md" not in existing:
        print("Uploading README.md (model card)...")
        api.upload_file(
            path_or_fileobj=README.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            commit_message="Add model card",
        )
    else:
        print("SKIP README.md (already on hub)")

    # 3. Upload eval_summary.json
    eval_json = RESULTS / LOCAL_CAMPAIGN_DIR / "eval_summary.json"
    if eval_json.exists() and f"{REMOTE_PREFIX}/eval_summary.json" not in existing:
        print(f"Uploading {REMOTE_PREFIX}/eval_summary.json...")
        api.upload_file(
            path_or_fileobj=str(eval_json),
            path_in_repo=f"{REMOTE_PREFIX}/eval_summary.json",
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            commit_message="Add multiseed-campaign eval summary",
        )
    else:
        print(f"SKIP {REMOTE_PREFIX}/eval_summary.json (already on hub or not found)")

    # 4. Upload checkpoints one by one (each is ~3-4 GB)
    # Reuse the existing-files set built above to skip already-uploaded files.
    for variant, seed in PHASE_D_SEEDS:
        local_path = RESULTS / LOCAL_CAMPAIGN_DIR / f"{variant}_{seed}" / "best_model.pt"
        remote_path = f"{REMOTE_PREFIX}/{variant}_{seed}/best_model.pt"

        if not local_path.exists():
            print(f"  SKIP (not found locally): {local_path}")
            continue

        if remote_path in existing:
            print(f"  SKIP (already on hub): {remote_path}")
            continue

        size_gb = local_path.stat().st_size / 1e9
        print(f"Uploading {remote_path}  ({size_gb:.2f} GB)...")
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=remote_path,
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            commit_message=f"Add {variant} {seed} checkpoint",
        )
        print(f"  Done: {remote_path}")

    print("\nAll uploads complete.")
    print(f"View at: https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
