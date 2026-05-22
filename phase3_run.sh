#!/usr/bin/env bash
# =============================================================================
# Phase 3 — Complete Retraining Experiments
# RTX 4090 / 24 GB VRAM / CUDA 12.6 / Python 3.12
#
# Run order:
#   R4+R2  : Morphaware FM2+FM1 fix (residual zero-init + val_loss selection)
#   R2     : Baseline FM1 fix (val_loss selection)
#   R15    : Morphaware λ=0.1 (ablation best)
#   R1     : Filter trivial pairs, retrain both models
#   R34    : Augmentation ablation (augment=0.1,0.2,0.3)
#   R6     : 6 missing individual-feature ablations
#   R3     : Aya-Expanse-8B zero-shot LLM baseline
#
# Usage:
#   screen -S phase3 bash /workspace/sorani-gec/phase3_run.sh 2>&1 | tee /workspace/phase3_master.log
# =============================================================================
set -uo pipefail

WORKDIR="/workspace/sorani-gec"
LOG_DIR="/workspace/phase3_logs"
RESULTS_DIR="$WORKDIR/results"
DATA_DIR="$WORKDIR/data/splits"
DATA_FILT_DIR="$WORKDIR/data/splits_filtered"

mkdir -p "$LOG_DIR" "$RESULTS_DIR"

cd "$WORKDIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
FAILED_EXPERIMENTS=()

run_experiment() {
    local name="$1"; shift
    log "=== START: $name ==="
    local start_ts=$SECONDS
    python3 "$@" 2>&1 | tee "$LOG_DIR/${name}.log"
    local rc=${PIPESTATUS[0]}
    local elapsed=$(( SECONDS - start_ts ))
    if [ $rc -eq 0 ]; then
        log "=== OK: $name (${elapsed}s) ==="
    else
        log "=== FAILED: $name (exit=$rc, ${elapsed}s) ==="
        FAILED_EXPERIMENTS+=("$name")
    fi
    return $rc
}

# ---------------------------------------------------------------------------
# STEP 0 — Sanity-check CUDA
# ---------------------------------------------------------------------------
log "Checking CUDA..."
python3 -c "import torch; assert torch.cuda.is_available(), 'No CUDA'; print('CUDA OK:', torch.cuda.get_device_name(0))"

# ---------------------------------------------------------------------------
# STEP 1 — R4+R2: Morphaware FM2 fix (residual zero-init) + val_loss selection
# ---------------------------------------------------------------------------
run_experiment "R4_R2_morphaware_v2" \
    scripts/06_train_morphaware.py \
    --selection-metric val_loss \
    --output-dir results/models/morphaware_v2 \
    --epochs 30 \
    --batch-size 16 \
    --grad-accum-steps 8 \
    --max-length 256 \
    --seed 42 \
    --agreement-loss-weight 0.3 \
    --eval-every-n-steps 0

# ---------------------------------------------------------------------------
# STEP 2 — R2: Baseline val_loss selection
# ---------------------------------------------------------------------------
run_experiment "R2_baseline_v2" \
    scripts/05_train_baseline.py \
    --selection-metric val_loss \
    --output-dir results/models/baseline_v2 \
    --epochs 30 \
    --batch-size 16 \
    --grad-accum-steps 8 \
    --max-length 256 \
    --seed 42 \
    --eval-every-n-steps 0

# ---------------------------------------------------------------------------
# STEP 3 — R15: Morphaware λ=0.1 + val_loss selection
# ---------------------------------------------------------------------------
run_experiment "R15_morphaware_lambda01" \
    scripts/06_train_morphaware.py \
    --selection-metric val_loss \
    --output-dir results/models/morphaware_lambda01 \
    --epochs 30 \
    --batch-size 16 \
    --grad-accum-steps 8 \
    --max-length 256 \
    --seed 42 \
    --agreement-loss-weight 0.1 \
    --eval-every-n-steps 0

# ---------------------------------------------------------------------------
# STEP 4 — R1: Create filtered splits (source != target only)
# ---------------------------------------------------------------------------
log "=== R1: Filtering trivial pairs ==="
python3 scripts/filter_trivial.py \
    --input data/splits \
    --output data/splits_filtered 2>&1 | tee "$LOG_DIR/R1_filter_trivial.log"

# ---------------------------------------------------------------------------
# STEP 5 — R1: Retrain baseline on filtered splits
# ---------------------------------------------------------------------------
run_experiment "R1_baseline_filtered" \
    scripts/05_train_baseline.py \
    --selection-metric val_loss \
    --data-dir data/splits_filtered \
    --output-dir results/models/baseline_filtered \
    --epochs 30 \
    --batch-size 16 \
    --grad-accum-steps 8 \
    --max-length 256 \
    --seed 42 \
    --eval-every-n-steps 0

# ---------------------------------------------------------------------------
# STEP 6 — R1: Retrain morphaware on filtered splits (λ=0.1, FM2 fix)
# ---------------------------------------------------------------------------
run_experiment "R1_morphaware_filtered" \
    scripts/06_train_morphaware.py \
    --selection-metric val_loss \
    --data-dir data/splits_filtered \
    --output-dir results/models/morphaware_filtered \
    --epochs 30 \
    --batch-size 16 \
    --grad-accum-steps 8 \
    --max-length 256 \
    --seed 42 \
    --agreement-loss-weight 0.1 \
    --eval-every-n-steps 0

# ---------------------------------------------------------------------------
# STEP 7 — R34: Augmentation ablation (three levels)
# ---------------------------------------------------------------------------
for AUG_RATIO in 0.1 0.2 0.3; do
    AUG_TAG=$(echo "$AUG_RATIO" | tr '.' '_')
    run_experiment "R34_augment_${AUG_TAG}" \
        scripts/06_train_morphaware.py \
        --selection-metric val_loss \
        --output-dir "results/models/augment_${AUG_TAG}" \
        --epochs 30 \
        --batch-size 16 \
        --grad-accum-steps 8 \
        --max-length 256 \
        --seed 42 \
        --agreement-loss-weight 0.1 \
        --augment "$AUG_RATIO" \
        --eval-every-n-steps 0
done

# ---------------------------------------------------------------------------
# STEP 8 — R6: Individual feature ablations (6 missing features)
# ---------------------------------------------------------------------------
run_experiment "R6_individual_features" \
    scripts/08_ablation.py \
    --experiment individual_features \
    --features aspect case definiteness transitivity clitic_person clitic_number \
    --output results/ablation \
    --test-data data/splits/test.jsonl

# ---------------------------------------------------------------------------
# STEP 9 — R3: Evaluate all models on test set (full + edited subset)
# ---------------------------------------------------------------------------
log "=== R3-prep: Generating evaluation results for all Phase 3 models ==="
python3 scripts/phase3_evaluate_all.py 2>&1 | tee "$LOG_DIR/R3_prep_evaluate.log"

# ---------------------------------------------------------------------------
# STEP 10 — R3: Aya-Expanse-8B zero-shot LLM baseline
# ---------------------------------------------------------------------------
log "=== R3: Aya-Expanse-8B zero-shot evaluation ==="
run_experiment "R3_aya_expanse_8b" \
    scripts/11_evaluate_llm_baseline.py \
    --test-data data/splits/test.jsonl \
    --edited-only \
    --output results/llm_baseline \
    --model CohereForAI/aya-expanse-8b \
    --batch-size 2 \
    --max-new-tokens 300

# ---------------------------------------------------------------------------
# DONE
# ---------------------------------------------------------------------------
log "=== Phase 3 complete ==="
log "Results in: $RESULTS_DIR"
log "Logs in:    $LOG_DIR"

if [ ${#FAILED_EXPERIMENTS[@]} -gt 0 ]; then
    log "FAILED experiments: ${FAILED_EXPERIMENTS[*]}"
else
    log "All experiments PASSED"
fi

# Print final summary
python3 scripts/phase3_summary.py 2>&1 | tee "$LOG_DIR/phase3_summary.log"
