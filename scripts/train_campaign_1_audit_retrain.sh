#!/bin/bash
# =============================================================================
# Campaign 1 — audit retrain (formerly "phase 1")
# Convergence + deconfounded morphaware (audit 2026-05-30)
# =============================================================================
# One-command relaunch of the two headline models with the campaign-1 fixes from
# notes/thesis_critical_audit_2026-05-30.md. It is GPU-bound: run it on the
# rental RTX 5090 (vast.ai), not on a CPU box.
#
# What changes versus train_campaign_2_multiseed.sh (the released runs):
#   1.1  --warmup-steps 60   ByT5/5k corpus has ~1260 optimiser steps total;
#                            the released warmup_steps=1000 pushed peak LR to
#                            ~epoch 24/30 (L8-01). 60 steps ≈ 1.4 epochs.
#   1.2  --selection-metric val_f05   Now that warmup is short, val-F0.5 stops
#                            spiking at epoch 1, so it is a usable selection
#                            signal again (L8-04/L8-05). Patience is raised so
#                            training runs to a real F0.5 plateau, not max_epochs.
#   1.3  --agreement-loss-weight 0.0  The ablation found lambda=0 is optimal, so
#                            the deconfounded morphaware run drops the agreement
#                            loss term entirely; this isolates the architecture
#                            from the auxiliary-loss confound behind the negative
#                            result (L9-02/L8-02).
#
# Seeds: 42 123 777 (the released set; see L9-03). Eval uses scripts/eval_campaign_checkpoints.py
# (evaluate_corpus, beam 4) — the same scorer the trainer uses for val-F0.5, so
# dev and test are scored identically (L8-06/1.5 verification).
#
# Prerequisites:
#   cd Implementation/sorani-gec
#   source .venv/bin/activate
#   mkdir -p results/campaign_1_audit_retrain
#
# Usage:
#   bash scripts/train_campaign_1_audit_retrain.sh                 # 3 seeds, both models
#   bash scripts/train_campaign_1_audit_retrain.sh --seeds "42"    # single-seed smoke test
#   bash scripts/train_campaign_1_audit_retrain.sh --baseline-only
#   bash scripts/train_campaign_1_audit_retrain.sh --morph-only
#
# Exit gate:
#   Six checkpoints under results/campaign_1_audit_retrain/, each selected on val-F0.5 plateau;
#   pooled 3-seed bootstrap rerun (scripts/run_bootstrap.py) on the new
#   hypotheses to refresh the headline gap.
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
PATIENCE=8            # raised from 5: let val-F0.5 reach a real plateau (1.2)
MAX_LEN=256
WARMUP=60            # 1.1: ~1.4 epochs of the ~1260-step schedule
AGR_WEIGHT=0.0       # 1.3: lambda=0, ablation-optimal, deconfounds the negative result
SELECTION="val_f05"  # 1.2: F0.5 plateau, not val_loss
RESULTS_DIR="results/campaign_1_audit_retrain"
RUN_BASELINE=true
RUN_MORPHAWARE=true

# ---- argument parsing ----
while [[ $# -gt 0 ]]; do
  case "$1" in
    --seeds)          SEEDS="$2"; shift 2 ;;
    --data-dir)       DATA_DIR="$2"; shift 2 ;;
    --results-dir)    RESULTS_DIR="$2"; shift 2 ;;
    --warmup-steps)   WARMUP="$2"; shift 2 ;;
    --agr-weight)     AGR_WEIGHT="$2"; shift 2 ;;
    --baseline-only)  RUN_MORPHAWARE=false; shift ;;
    --morph-only)     RUN_BASELINE=false; shift ;;
    --help|-h)
      grep '^#' "$0" | head -50; exit 0 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

echo "==================================================================="
echo "Campaign 1 audit retrain (convergence + deconfounded morphaware)"
echo "  Seeds:        $SEEDS"
echo "  Data:         $DATA_DIR"
echo "  Results:      $RESULTS_DIR"
echo "  Warmup:       $WARMUP steps"
echo "  Selection:    $SELECTION"
echo "  Agr weight:   $AGR_WEIGHT (morphaware)"
echo "  Patience:     $PATIENCE"
echo "  Baseline:     $RUN_BASELINE"
echo "  Morphaware:   $RUN_MORPHAWARE"
echo "  Started:      $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
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

    python3 scripts/05_train_baseline.py \
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
      --warmup-steps "$WARMUP" \
      --selection-metric "$SELECTION" \
      2>&1 | tee "$LOG"

    echo "    [DONE] baseline seed=$SEED at $(date -u '+%H:%M:%S UTC')"
  fi

  # ---- morphology-aware (gated-residual ARCH-9, lambda=0) ----
  if [[ "$RUN_MORPHAWARE" == "true" ]]; then
    OUT_DIR="$RESULTS_DIR/morphaware_seed${SEED}"
    LOG="$RESULTS_DIR/morphaware_seed${SEED}.log"
    echo ""
    echo "--- MORPHAWARE (gated-residual, lambda=$AGR_WEIGHT) seed=$SEED ---"
    echo "    Output: $OUT_DIR"
    echo "    Log:    $LOG"

    python3 scripts/06_train_morphaware.py \
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
      --warmup-steps "$WARMUP" \
      --agreement-loss-weight "$AGR_WEIGHT" \
      --selection-metric "$SELECTION" \
      2>&1 | tee "$LOG"

    echo "    [DONE] morphaware seed=$SEED at $(date -u '+%H:%M:%S UTC')"
  fi

done  # seeds loop

echo ""
echo "==================================================================="
echo "Campaign 1 audit retrain complete — $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "Next: evaluate each checkpoint with the unified scorer, then rerun bootstrap."
echo ""
echo "  # both eval scripts honour CAMPAIGN_RESULTS_DIR / CAMPAIGN_DATA_DIR"
echo "  export CAMPAIGN_RESULTS_DIR=$RESULTS_DIR"
echo "  export CAMPAIGN_DATA_DIR=$DATA_DIR"
echo "  python scripts/eval_campaign_checkpoints.py   # aggregate F0.5 mean+/-std per model"
echo "  python scripts/dump_hypotheses.py     # write per-seed hypotheses.jsonl"
echo ""
echo "  # point run_bootstrap.py at the new hypotheses (CAMPAIGN_DIR env var), then:"
echo "  python scripts/run_bootstrap.py        # refresh pooled 3-seed headline gap"
echo "==================================================================="
