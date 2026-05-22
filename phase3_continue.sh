#!/usr/bin/env bash
# =============================================================================
# Phase 3 Continuation — Runs AFTER morphaware_p3 finishes
#
# Already done (previous sessions):
#   - baseline_p3         (R2: baseline FM1 fix, val_loss)
#   - morphaware_p3       (R4+R2: morphaware FM2+FM1 fix, λ=0.3)  [still running]
#
# This script runs:
#   - R15: morphaware_lambda01 (λ=0.1)
#   - R1:  baseline_filtered, morphaware_filtered (on filtered splits)
#   - R34: augment ablations (0.1, 0.2, 0.3)
#   - R6:  individual feature ablations
#   - R3:  Aya-Expanse-8B zero-shot LLM baseline
#   - Final evaluation of all models
#
# Usage:
#   screen -S phase3b bash /workspace/sorani-gec/phase3_continue.sh 2>&1 | tee /workspace/phase3b_master.log
# =============================================================================
set -uo pipefail

WORKDIR="/workspace/sorani-gec"
LOG_DIR="/workspace/phase3_logs"
RESULTS_DIR="$WORKDIR/results"

mkdir -p "$LOG_DIR"
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
# STEP 0 — Wait for morphaware_p3 to finish
# ---------------------------------------------------------------------------
MORPHAWARE_P3_PIDS=(3663 3846 3847 3848 3849)
log "Waiting for morphaware_p3 training to complete (PIDs: ${MORPHAWARE_P3_PIDS[*]})..."

while true; do
    all_done=true
    for pid in "${MORPHAWARE_P3_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            all_done=false
            break
        fi
    done

    if $all_done; then
        log "morphaware_p3 training appears finished."
        break
    fi

    # Also check if best_model.pt exists
    if [ -f "$WORKDIR/results/models/morphaware_p3/best_model.pt" ]; then
        # Check if the training is still running (model might be checkpointed but training continues)
        still_running=false
        for pid in "${MORPHAWARE_P3_PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                still_running=true
                break
            fi
        done
        if ! $still_running; then
            log "morphaware_p3 checkpoint exists and process is done."
            break
        fi
    fi

    log "Still waiting... GPU processes active. Sleeping 120s."
    sleep 120
done

# Verify GPU is free
log "Checking GPU memory..."
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader
python3 -c "import torch; print('CUDA:', torch.cuda.get_device_name(0)); print('Free:', round(torch.cuda.mem_get_info()[0]/1e9, 2), 'GB')"

# ---------------------------------------------------------------------------
# STEP 1 — R2: Baseline retrain with fp16=False + val_loss selection
# ---------------------------------------------------------------------------
run_experiment "R2_baseline_v2" \
    scripts/05_train_baseline.py \
    --selection-metric val_loss \
    --output-dir results/models/baseline_v2 \
    --epochs 15 \
    --batch-size 16 \
    --grad-accum-steps 4 \
    --max-length 256 \
    --seed 42 \
    --patience 4

# ---------------------------------------------------------------------------
# STEP 2 — R15: Morphaware λ=0.1
# ---------------------------------------------------------------------------
run_experiment "R15_morphaware_lambda01" \
    scripts/06_train_morphaware.py \
    --selection-metric val_loss \
    --output-dir results/models/morphaware_lambda01 \
    --epochs 15 \
    --batch-size 16 \
    --grad-accum-steps 4 \
    --max-length 256 \
    --seed 42 \
    --patience 4 \
    --agreement-loss-weight 0.1

# ---------------------------------------------------------------------------
# STEP 3 — R1: Baseline on filtered splits (source != target only)
# ---------------------------------------------------------------------------
run_experiment "R1_baseline_filtered" \
    scripts/05_train_baseline.py \
    --selection-metric val_loss \
    --data-dir data/splits_filtered \
    --output-dir results/models/baseline_filtered \
    --epochs 15 \
    --batch-size 16 \
    --grad-accum-steps 4 \
    --max-length 256 \
    --seed 42 \
    --patience 4

# ---------------------------------------------------------------------------
# STEP 3 — R1: Morphaware on filtered splits (λ=0.1)
# ---------------------------------------------------------------------------
run_experiment "R1_morphaware_filtered" \
    scripts/06_train_morphaware.py \
    --selection-metric val_loss \
    --data-dir data/splits_filtered \
    --output-dir results/models/morphaware_filtered \
    --epochs 15 \
    --batch-size 16 \
    --grad-accum-steps 4 \
    --max-length 256 \
    --seed 42 \
    --patience 4 \
    --agreement-loss-weight 0.1

# ---------------------------------------------------------------------------
# STEP 4 — R34: Augmentation ablation (three levels)
# ---------------------------------------------------------------------------
for AUG_RATIO in 0.1 0.2 0.3; do
    AUG_TAG=$(echo "$AUG_RATIO" | tr '.' '_')
    run_experiment "R34_augment_${AUG_TAG}" \
        scripts/06_train_morphaware.py \
        --selection-metric val_loss \
        --output-dir "results/models/augment_${AUG_TAG}" \
        --epochs 15 \
        --batch-size 16 \
        --grad-accum-steps 4 \
        --max-length 256 \
        --seed 42 \
        --patience 4 \
        --agreement-loss-weight 0.1 \
        --augment "$AUG_RATIO"
done

# ---------------------------------------------------------------------------
# STEP 5 — R6: Individual feature ablations (6 missing features)
# ---------------------------------------------------------------------------
run_experiment "R6_individual_features" \
    scripts/08_ablation.py \
    --experiment individual_features \
    --features aspect case definiteness transitivity clitic_person clitic_number \
    --output results/ablation \
    --test-data data/splits/test.jsonl

# ---------------------------------------------------------------------------
# STEP 6 — R3: Evaluate all trained models (baseline_p3 + morphaware_p3 + new)
# ---------------------------------------------------------------------------
log "=== R3-prep: Evaluating all Phase 3 models ==="
python3 scripts/phase3_evaluate_all.py 2>&1 | tee "$LOG_DIR/phase3_evaluate.log"

# ---------------------------------------------------------------------------
# STEP 7 — R3: Aya-Expanse-8B zero-shot LLM baseline
# ---------------------------------------------------------------------------
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
log "=== Phase 3 Continuation complete ==="
if [ ${#FAILED_EXPERIMENTS[@]} -gt 0 ]; then
    log "FAILED: ${FAILED_EXPERIMENTS[*]}"
else
    log "All experiments PASSED"
fi

python3 scripts/phase3_summary.py 2>&1 | tee "$LOG_DIR/phase3b_summary.log"
