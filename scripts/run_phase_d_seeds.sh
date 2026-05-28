#!/bin/bash
# =============================================================================
# Phase D multi-seed training — R2 (gated-residual morphaware) + R32 (≥3 seeds)
# =============================================================================
# Trains the ByT5-small baseline and the gated-residual morphology-aware model
# (ARCH-9, morph_gate parameter, zero-init morph_residual_proj) with three
# independent random seeds on splits_v2 data.
#
# Selection criterion: val_loss (not val_f05 — see R4 fix in configs/default.yaml).
# FP16: disabled (ByT5 byte-level embeddings produce NaN on CUDA 12.x with FP16).
#
# Prerequisites:
#   cd Implementation/sorani-gec
#   source .venv/bin/activate   # or conda activate sorani-gec
#   mkdir -p results/phase_d
#
# Usage:
#   bash scripts/run_phase_d_seeds.sh                  # runs 3 seeds both models
#   bash scripts/run_phase_d_seeds.sh --seeds "42"     # single seed smoke test
#   bash scripts/run_phase_d_seeds.sh --baseline-only  # skip morphaware
#   bash scripts/run_phase_d_seeds.sh --morph-only     # skip baseline
#
# Exit gate (Chapter 7 §7.4):
#   All six checkpoints saved; mean±std tables in Ch 7 filled from eval logs.
#   Required: F0.5 (edited) > 0.01 for at least two seeds (trivial non-zero signal).
# =============================================================================

set -euo pipefail

# ---- defaults ----
SEEDS="42 123 777"
DATA_DIR="data/splits_v2"
MODEL="google/byt5-small"
LR="5e-5"
BATCH=16
ACCUM=8
EPOCHS=30
PATIENCE=5
MAX_LEN=256
RESULTS_DIR="results/phase_d"
RUN_BASELINE=true
RUN_MORPHAWARE=true

# ---- argument parsing ----
while [[ $# -gt 0 ]]; do
  case "$1" in
    --seeds)          SEEDS="$2"; shift 2 ;;
    --data-dir)       DATA_DIR="$2"; shift 2 ;;
    --results-dir)    RESULTS_DIR="$2"; shift 2 ;;
    --baseline-only)  RUN_MORPHAWARE=false; shift ;;
    --morph-only)     RUN_BASELINE=false; shift ;;
    --help|-h)
      grep '^#' "$0" | head -40; exit 0 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

echo "==================================================================="
echo "Phase D multi-seed training (R2 + R32)"
echo "  Seeds:      $SEEDS"
echo "  Data:       $DATA_DIR"
echo "  Results:    $RESULTS_DIR"
echo "  Baseline:   $RUN_BASELINE"
echo "  Morphaware: $RUN_MORPHAWARE"
echo "  Started:    $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "==================================================================="

mkdir -p "$RESULTS_DIR"

for SEED in $SEEDS; do

  # ---- baseline (ByT5-small, no morphology) ----
  if [[ "$RUN_BASELINE" == "true" ]]; then
    OUT_DIR="$RESULTS_DIR/baseline_seed${SEED}"
    LOG="$RESULTS_DIR/baseline_seed${SEED}.log"
    echo ""
    echo "--- BASELINE seed=$SEED ---"
    echo "    Output: $OUT_DIR"
    echo "    Log:    $LOG"

    python scripts/05_train_baseline.py \
      --data-dir "$DATA_DIR" \
      --output-dir "$OUT_DIR" \
      --model "$MODEL" \
      --seed "$SEED" \
      --lr "$LR" \
      --batch-size "$BATCH" \
      --grad-accum-steps "$ACCUM" \
      --epochs "$EPOCHS" \
      --patience "$PATIENCE" \
      --max-length "$MAX_LEN" \
      --selection-metric val_loss \
      2>&1 | tee "$LOG"

    echo "    [DONE] baseline seed=$SEED at $(date -u '+%H:%M:%S UTC')"
  fi

  # ---- morphology-aware (gated-residual ARCH-9) ----
  if [[ "$RUN_MORPHAWARE" == "true" ]]; then
    OUT_DIR="$RESULTS_DIR/morphaware_seed${SEED}"
    LOG="$RESULTS_DIR/morphaware_seed${SEED}.log"
    echo ""
    echo "--- MORPHAWARE (gated-residual) seed=$SEED ---"
    echo "    Output: $OUT_DIR"
    echo "    Log:    $LOG"

    python scripts/06_train_morphaware.py \
      --data-dir "$DATA_DIR" \
      --output-dir "$OUT_DIR" \
      --model "$MODEL" \
      --seed "$SEED" \
      --lr "$LR" \
      --batch-size "$BATCH" \
      --grad-accum-steps "$ACCUM" \
      --epochs "$EPOCHS" \
      --patience "$PATIENCE" \
      --max-length "$MAX_LEN" \
      --selection-metric val_loss \
      2>&1 | tee "$LOG"

    echo "    [DONE] morphaware seed=$SEED at $(date -u '+%H:%M:%S UTC')"
  fi

done  # seeds loop

echo ""
echo "==================================================================="
echo "Phase D training complete — $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "Next: run evaluation on each checkpoint."
echo ""
echo "  for SEED in $SEEDS; do"
echo "    python scripts/07_evaluate.py \\"
echo "      --model-dir $RESULTS_DIR/baseline_seed\${SEED} \\"
echo "      --data-dir  $DATA_DIR/test.jsonl \\"
echo "      --out       $RESULTS_DIR/baseline_seed\${SEED}_eval.json"
echo ""
echo "    python scripts/07_evaluate.py \\"
echo "      --model-dir $RESULTS_DIR/morphaware_seed\${SEED} \\"
echo "      --data-dir  $DATA_DIR/test.jsonl \\"
echo "      --out       $RESULTS_DIR/morphaware_seed\${SEED}_eval.json"
echo "  done"
echo ""
echo "Then fill Tab 7.3 / Tab 7.4 in research_thesis/chapters/07_results.tex"
echo "with mean±std values from the six _eval.json files."
echo "==================================================================="
