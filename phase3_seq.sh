#!/usr/bin/env bash
# Sequential Phase 3 — clean run in correct order
# Baseline (R2) first, then morphaware (R4+R2), then remaining experiments
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

# ---- R2: Baseline retrain (fp16=False via fixed config) ----
run_exp "R2_baseline_p3" \
    scripts/05_train_baseline.py \
    --selection-metric val_loss \
    --output-dir results/models/baseline_p3 \
    --epochs 15 --batch-size 16 --grad-accum-steps 4 \
    --max-length 256 --seed 42 --patience 4

# ---- R4+R2: Morphaware FM2+FM1 fix (λ=0.3) ----
run_exp "R4_R2_morphaware_p3" \
    scripts/06_train_morphaware.py \
    --selection-metric val_loss \
    --output-dir results/models/morphaware_p3 \
    --epochs 15 --batch-size 16 --grad-accum-steps 4 \
    --max-length 256 --seed 42 --patience 4 \
    --agreement-loss-weight 0.3

# ---- R15: Morphaware λ=0.1 ----
run_exp "R15_morphaware_lambda01" \
    scripts/06_train_morphaware.py \
    --selection-metric val_loss \
    --output-dir results/models/morphaware_lambda01 \
    --epochs 15 --batch-size 16 --grad-accum-steps 4 \
    --max-length 256 --seed 42 --patience 4 \
    --agreement-loss-weight 0.1

# ---- R1: Baseline on filtered splits (source != target) ----
run_exp "R1_baseline_filtered" \
    scripts/05_train_baseline.py \
    --selection-metric val_loss \
    --data-dir data/splits_filtered \
    --output-dir results/models/baseline_filtered \
    --epochs 15 --batch-size 16 --grad-accum-steps 4 \
    --max-length 256 --seed 42 --patience 4

# ---- R1: Morphaware on filtered splits (λ=0.1) ----
run_exp "R1_morphaware_filtered" \
    scripts/06_train_morphaware.py \
    --selection-metric val_loss \
    --data-dir data/splits_filtered \
    --output-dir results/models/morphaware_filtered \
    --epochs 15 --batch-size 16 --grad-accum-steps 4 \
    --max-length 256 --seed 42 --patience 4 \
    --agreement-loss-weight 0.1

# ---- R34: Augmentation ablation (3 levels) ----
for AUG in 0.1 0.2 0.3; do
    TAG=$(echo "$AUG" | tr '.' '_')
    run_exp "R34_augment_${TAG}" \
        scripts/06_train_morphaware.py \
        --selection-metric val_loss \
        --output-dir "results/models/augment_${TAG}" \
        --epochs 15 --batch-size 16 --grad-accum-steps 4 \
        --max-length 256 --seed 42 --patience 4 \
        --agreement-loss-weight 0.1 --augment "$AUG"
done

# ---- R6: Individual feature ablations (6 missing features) ----
run_exp "R6_individual_features" \
    scripts/08_ablation.py \
    --experiment individual_features \
    --features aspect case definiteness transitivity clitic_person clitic_number \
    --output results/ablation \
    --test-data data/splits/test.jsonl

# ---- Evaluate all models ----
log "=== Evaluating all Phase 3 models ==="
python3 scripts/phase3_evaluate_all.py 2>&1 | tee "$LOG_DIR/phase3_evaluate.log"

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

log "=== All done. Failed: ${FAILED[*]:-none} ==="
