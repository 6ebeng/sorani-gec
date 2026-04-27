#!/bin/bash
# Phase 2 retrain: FM1 (val_loss selection) + FM2 (residual zero-init morph) — 27 Apr 2026
# Use a non-existent config path to keep CLI defaults (fp16=False; ByT5 NaN's in fp16).
set -euo pipefail
cd /workspace/sorani-gec
. .venv/bin/activate
mkdir -p results/models/baseline_valloss results/models/morphaware_valloss

echo "=== BASELINE val_loss (FP32) ==="
python scripts/05_train_baseline.py \
  --config /tmp/no_such_config.yaml \
  --epochs 30 --batch-size 16 --grad-accum-steps 8 --lr 5e-5 \
  --model google/byt5-small --data-dir data/splits \
  --output-dir results/models/baseline_valloss --max-length 256 \
  --patience 5 --selection-metric val_loss \
  2>&1 | tee results/baseline_valloss.log

echo "=== MORPHAWARE val_loss (FP32) ==="
python scripts/06_train_morphaware.py \
  --config /tmp/no_such_config.yaml \
  --epochs 30 --batch-size 16 --grad-accum-steps 8 --lr 5e-5 \
  --model google/byt5-small --data-dir data/splits \
  --output-dir results/models/morphaware_valloss --max-length 256 \
  --patience 5 --selection-metric val_loss \
  2>&1 | tee results/morphaware_valloss.log

echo "=== DONE ==="

