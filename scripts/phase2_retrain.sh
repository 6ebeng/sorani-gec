#!/bin/bash
# =============================================================================
# Phase 2 retrain — scaled data + warm-started morphology injection
# =============================================================================
# One-command relaunch for the rented RTX 5090 (vast.ai / Linux). Builds on
# phase1_retrain.sh but adds the two levers phase 1 left on the table:
#
#   2.1  Scaled training data. splits_v2 train is only 5,253 single-edit pairs;
#        a 300M ByT5 starves on that. scripts/14_build_scaled_train.py lifts it
#        ~5-7x from the 50k-sentence balanced corpus, keeping single-edit
#        discipline and dedup vs the FIXED dev/test (test stays byte-identical).
#   2.2  Warm-started morph injection. The released gated-residual starts with
#        gate=0 and a zero-init residual, so morphology is inert at step 1 and
#        has to climb out of a zero-gradient identity. --morph-gate-init 0.5 and
#        --morph-residual-init 0.02 let morph features fire from the first step.
#
# Carried over from phase 1 (the convergence fixes):
#   - --warmup-steps sized to ~1.5 epochs of the NEW step count (not 1000);
#   - --selection-metric val_f05 (usable once warmup is short);
#   - --agreement-loss-weight 0.0 (ablation-optimal; deconfounds the result).
#
# Honest expectation: the warmup + data fixes should lift BOTH models well off
# the F0.5~0.08 floor (a working baseline is the high-probability win). Whether
# morph then beats baseline is a genuine open question — this run measures it,
# it does not assume it.
#
# Seeds: 42 123 777. Results -> results/phase2/, scored with the same pipeline
# as phase 1 (eval_phase_d.py + dump_hypotheses.py + run_bootstrap.py).
#
# Prerequisites:
#   cd Implementation/sorani-gec && source .venv/bin/activate
#
# Usage:
#   bash scripts/phase2_retrain.sh                   # build data + 3 seeds, both
#   bash scripts/phase2_retrain.sh --seeds "42"      # single-seed smoke
#   bash scripts/phase2_retrain.sh --skip-build      # reuse data/splits_scaled
#   bash scripts/phase2_retrain.sh --baseline-only
#   bash scripts/phase2_retrain.sh --morph-only
# =============================================================================

set -euo pipefail

# ---- defaults ----
SEEDS="42 123 777"
DATA_DIR="data/splits_scaled"
POOL_TARGET=50000
AGR_OVERSAMPLE=1.5
MODEL="google/byt5-small"
LR="5e-5"
BATCH=16
ACCUM=8
EPOCHS=30
PATIENCE=8
MAX_LEN=256
AGR_WEIGHT=0.0          # 1.3: lambda=0, ablation-optimal
SELECTION="val_f05"
GATE_INIT=0.5           # 2.2: open the morph gate from step 1
RESIDUAL_INIT=0.02      # 2.2: small-normal residual seed
RESULTS_DIR="results/phase2"
RUN_BASELINE=true
RUN_MORPHAWARE=true
SKIP_BUILD=false

# ---- argument parsing ----
while [[ $# -gt 0 ]]; do
  case "$1" in
    --seeds)          SEEDS="$2"; shift 2 ;;
    --batch)          BATCH="$2"; shift 2 ;;
    --accum)          ACCUM="$2"; shift 2 ;;
    --max-len)        MAX_LEN="$2"; shift 2 ;;
    --data-dir)       DATA_DIR="$2"; shift 2 ;;
    --results-dir)    RESULTS_DIR="$2"; shift 2 ;;
    --pool-target)    POOL_TARGET="$2"; shift 2 ;;
    --agr-oversample) AGR_OVERSAMPLE="$2"; shift 2 ;;
    --gate-init)      GATE_INIT="$2"; shift 2 ;;
    --residual-init)  RESIDUAL_INIT="$2"; shift 2 ;;
    --agr-weight)     AGR_WEIGHT="$2"; shift 2 ;;
    --skip-build)     SKIP_BUILD=true; shift ;;
    --baseline-only)  RUN_MORPHAWARE=false; shift ;;
    --morph-only)     RUN_BASELINE=false; shift ;;
    --help|-h)        grep '^#' "$0" | head -55; exit 0 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

# ---- 1) build scaled data ----
if [[ "$SKIP_BUILD" != "true" ]]; then
  echo "=== Building scaled training data -> $DATA_DIR ==="
  python3 scripts/14_build_scaled_train.py \
    --target "$POOL_TARGET" \
    --agreement-oversample "$AGR_OVERSAMPLE" \
    --eval-dir data/splits_v2 \
    --output-dir "$DATA_DIR"
fi

if [[ ! -f "$DATA_DIR/train.jsonl" ]]; then
  echo "ERROR: $DATA_DIR/train.jsonl missing. Run without --skip-build."; exit 1
fi

# ---- 2) size warmup to ~1.5 epochs of the NEW step count ----
WARMUP=$(python3 - "$DATA_DIR/train.jsonl" "$BATCH" "$ACCUM" <<'PY'
import math, sys
train, batch, accum = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
n = sum(1 for _ in open(train, encoding="utf-8"))
steps_per_epoch = max(1, math.ceil(n / (batch * accum)))
print(max(50, round(1.5 * steps_per_epoch)))
PY
)
TRAIN_N=$(wc -l < "$DATA_DIR/train.jsonl")

echo "==================================================================="
echo "Phase 2 retrain (scaled data + warm-started morphology)"
echo "  Seeds:        $SEEDS"
echo "  Data:         $DATA_DIR  (train=$TRAIN_N pairs)"
echo "  Results:      $RESULTS_DIR"
echo "  Warmup:       $WARMUP steps (~1.5 epochs)"
echo "  Selection:    $SELECTION"
echo "  Agr weight:   $AGR_WEIGHT (morphaware)"
echo "  Gate init:    $GATE_INIT   Residual init: $RESIDUAL_INIT"
echo "  Oversample:   x$AGR_OVERSAMPLE core-agreement"
echo "  Started:      $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "==================================================================="

mkdir -p "$RESULTS_DIR"

for SEED in $SEEDS; do

  if [[ "$RUN_BASELINE" == "true" ]]; then
    OUT_DIR="$RESULTS_DIR/baseline_seed${SEED}"
    LOG="$RESULTS_DIR/baseline_seed${SEED}.log"
    echo ""; echo "--- BASELINE seed=$SEED -> $OUT_DIR ---"
    python3 scripts/05_train_baseline.py \
      --data-dir "$DATA_DIR" --output-dir "$OUT_DIR" --model "$MODEL" \
      --seed "$SEED" --lr "$LR" --batch-size "$BATCH" --grad-accum-steps "$ACCUM" \
      --epochs "$EPOCHS" --patience "$PATIENCE" --max-length "$MAX_LEN" \
      --warmup-steps "$WARMUP" --selection-metric "$SELECTION" \
      2>&1 | tee "$LOG"
    echo "    [DONE] baseline seed=$SEED at $(date -u '+%H:%M:%S UTC')"
  fi

  if [[ "$RUN_MORPHAWARE" == "true" ]]; then
    OUT_DIR="$RESULTS_DIR/morphaware_seed${SEED}"
    LOG="$RESULTS_DIR/morphaware_seed${SEED}.log"
    echo ""; echo "--- MORPHAWARE (gate=$GATE_INIT, lambda=$AGR_WEIGHT) seed=$SEED -> $OUT_DIR ---"
    python3 scripts/06_train_morphaware.py \
      --data-dir "$DATA_DIR" --output-dir "$OUT_DIR" --model "$MODEL" \
      --seed "$SEED" --lr "$LR" --batch-size "$BATCH" --grad-accum-steps "$ACCUM" \
      --epochs "$EPOCHS" --patience "$PATIENCE" --max-length "$MAX_LEN" \
      --warmup-steps "$WARMUP" --agreement-loss-weight "$AGR_WEIGHT" \
      --selection-metric "$SELECTION" \
      --morph-gate-init "$GATE_INIT" --morph-residual-init "$RESIDUAL_INIT" \
      2>&1 | tee "$LOG"
    echo "    [DONE] morphaware seed=$SEED at $(date -u '+%H:%M:%S UTC')"
  fi

done

# ---- 3) evaluate + dump hypotheses + bootstrap ----
echo ""; echo "=== Evaluation (PHASE_RESULTS_DIR=$RESULTS_DIR PHASE_DATA_DIR=$DATA_DIR) ==="
export PHASE_RESULTS_DIR="$RESULTS_DIR"
export PHASE_DATA_DIR="$DATA_DIR"
python3 scripts/eval_phase_d.py     | tee "$RESULTS_DIR/eval_summary.txt"
python3 scripts/dump_hypotheses.py
PHASE_D_DIR="$RESULTS_DIR" python3 scripts/run_bootstrap.py | tee "$RESULTS_DIR/bootstrap.txt"

echo ""
echo "==================================================================="
echo "Phase 2 complete — $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "  Headline gap: see $RESULTS_DIR/bootstrap.txt"
echo "  Per-model F0.5: see $RESULTS_DIR/eval_summary.txt"
echo "  Then locally: rerun scripts/13_agreement_subset_rescore.py pointed at"
echo "  $RESULTS_DIR to refresh the agreement-density table on the new models."
echo "==================================================================="
