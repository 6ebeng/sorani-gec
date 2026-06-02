#!/bin/bash
# Short confirmation run: baseline-only, 1 seed, 10 epochs, clean splits_scaled.
# Writes to results/phase2_confirm so the full sweep dir stays clean.
set -euo pipefail
cd /root/sorani-gec
source .venv/bin/activate
export PYTHONIOENCODING=utf-8

DATA_DIR=data/splits_scaled
OUT_DIR=results/phase2_confirm/baseline_seed42
mkdir -p results/phase2_confirm

# warmup ~1.5 epochs at batch32/accum4: n=26841 -> steps/epoch=210 -> warmup=315
python3 scripts/05_train_baseline.py \
  --data-dir "$DATA_DIR" --output-dir "$OUT_DIR" --model google/byt5-small \
  --seed 42 --lr 5e-5 --batch-size 32 --grad-accum-steps 4 \
  --epochs 10 --patience 6 --max-length 256 \
  --warmup-steps 315 --selection-metric val_f05

echo "CONFIRM_DONE $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
