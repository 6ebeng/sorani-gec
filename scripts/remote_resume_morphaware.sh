#!/bin/bash
# Phase 2 RESUME: morphaware from epoch 5 ckpt, with batched dev eval (8eca231)
set -euo pipefail
cd /workspace/sorani-gec
. .venv/bin/activate

CKPT=$(ls -t results/models/morphaware_valloss/checkpoint_epoch*.pt | head -1)
echo "=== Resuming from: $CKPT ==="

python scripts/06_train_morphaware.py \
  --config /tmp/no_such_config.yaml \
  --epochs 30 --batch-size 16 --grad-accum-steps 8 --lr 5e-5 \
  --model google/byt5-small --data-dir data/splits \
  --output-dir results/models/morphaware_valloss --max-length 256 \
  --patience 5 --selection-metric val_loss \
  --resume-from "$CKPT" \
  2>&1 | tee -a results/morphaware_valloss.log

echo "=== DONE ==="
