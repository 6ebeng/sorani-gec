#!/usr/bin/env bash
# Phase 3 Resume — picks up after augment_0_1 completed
# Restarts augment_0_2 from scratch (instance was interrupted at epoch 9)
set -uo pipefail
WORKDIR="/workspace/sorani-gec"
LOG_DIR="/workspace/phase3_logs"
mkdir -p "$LOG_DIR"
cd "$WORKDIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
FAILED=()

run_exp() {
    local name="$1"; shift
    log "=== START: $name ==="
    local t0=$SECONDS
    python3 "$@" 2>&1 | tee "$LOG_DIR/${name}.log"
    local rc=${PIPESTATUS[0]}
    local dt=$(( SECONDS - t0 ))
    if [ $rc -eq 0 ]; then
        log "=== OK: $name (${dt}s) ==="
    else
        log "=== FAILED: $name (rc=$rc ${dt}s) ==="
        FAILED+=("$name")
    fi
    return $rc
}

log "GPU state:"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader
python3 -c "import torch; print('CUDA:', torch.cuda.get_device_name(0))"
log "Resuming from augment_0_2 (restarting — instance interrupted at epoch 9)"

# ---- R34: augment_0_2 (restart from scratch, overwrites partial epoch-9 checkpoint) ----
run_exp "R34_augment_0_2" \
    scripts/06_train_morphaware.py \
    --selection-metric val_loss \
    --output-dir results/models/augment_0_2 \
    --epochs 15 --batch-size 16 --grad-accum-steps 4 \
    --max-length 256 --seed 42 --patience 4 \
    --agreement-loss-weight 0.1 --augment 0.2

# ---- R34: augment_0_3 ----
run_exp "R34_augment_0_3" \
    scripts/06_train_morphaware.py \
    --selection-metric val_loss \
    --output-dir results/models/augment_0_3 \
    --epochs 15 --batch-size 16 --grad-accum-steps 4 \
    --max-length 256 --seed 42 --patience 4 \
    --agreement-loss-weight 0.1 --augment 0.3

# ---- R6: Individual feature ablations ----
run_exp "R6_individual_features" \
    scripts/08_ablation.py \
    --experiment individual_features \
    --features aspect case definiteness transitivity clitic_person clitic_number \
    --output results/ablation \
    --test-data data/splits/test.jsonl

# ---- Evaluate all Phase 3 models ----
log "=== Evaluating all Phase 3 models ==="
python3 scripts/phase3_evaluate_all.py 2>&1 | tee "$LOG_DIR/phase3_evaluate.log"
eval_rc=${PIPESTATUS[0]}
if [ $eval_rc -ne 0 ]; then
    log "=== FAILED: phase3_evaluate_all (rc=$eval_rc) ==="
    FAILED+=("phase3_evaluate_all")
else
    log "=== OK: phase3_evaluate_all ==="
fi

# ---- R3: Aya-Expanse-8B zero-shot LLM baseline ----
run_exp "R3_aya_expanse_8b" \
    scripts/11_evaluate_llm_baseline.py \
    --test-data data/splits/test.jsonl \
    --edited-only \
    --output results/llm_baseline \
    --model CohereForAI/aya-expanse-8b \
    --batch-size 2 --max-new-tokens 300

# ---- Summary ----
python3 scripts/phase3_summary.py 2>&1 | tee "$LOG_DIR/phase3_summary.log"

log "=== Resume complete. Failed: ${FAILED[*]:-none} ==="
