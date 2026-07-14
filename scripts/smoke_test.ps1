# =============================================================================
# Local CPU smoke test for the scaled-retrain (campaign 3) pipeline (Windows / PowerShell).
# Verifies the WHOLE path runs end-to-end before you rent the RTX 5090:
#   1. build a tiny scaled split,
#   2. train the baseline for 1 epoch,
#   3. train the morphology-aware model for 1 epoch WITH the new gate-init flags,
#   4. confirm both produced a checkpoint.
# It proves the code works. It does NOT produce meaningful F0.5 (data is tiny).
#
# Usage:
#   $env:PYTHONIOENCODING = "utf-8"
#   .\scripts\smoke_test.ps1
# =============================================================================
$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
$py = "c:\Users\Tishko\Desktop\Thesis\.venv\Scripts\python.exe"

$DATA = "data\splits_smoke"
$OUT  = "results\smoke"

Write-Host "=== [1/4] Build tiny scaled split ($DATA) ===" -ForegroundColor Cyan
& $py scripts\14_build_scaled_train.py --target 600 `
    --pool-out data\synthetic_smoke --output-dir $DATA --agreement-oversample 1.5
if ($LASTEXITCODE -ne 0) { throw "data build failed" }

Write-Host "=== [2/4] Train baseline (1 epoch, CPU) ===" -ForegroundColor Cyan
& $py scripts\05_train_baseline.py --data-dir $DATA --output-dir "$OUT\baseline_seed42" `
    --model google/byt5-small --seed 42 --epochs 1 --batch-size 8 --grad-accum-steps 1 `
    --max-length 64 --warmup-steps 5 --selection-metric val_f05 --device cpu
if ($LASTEXITCODE -ne 0) { throw "baseline training failed" }

Write-Host "=== [3/4] Train morphaware (1 epoch, gate-init 0.5, lambda 0, CPU) ===" -ForegroundColor Cyan
& $py scripts\06_train_morphaware.py --data-dir $DATA --output-dir "$OUT\morphaware_seed42" `
    --model google/byt5-small --seed 42 --epochs 1 --batch-size 8 --grad-accum-steps 1 `
    --max-length 64 --warmup-steps 5 --selection-metric val_f05 `
    --agreement-loss-weight 0.0 --morph-gate-init 0.5 --morph-residual-init 0.02 --device cpu
if ($LASTEXITCODE -ne 0) { throw "morphaware training failed" }

Write-Host "=== [4/4] Verify checkpoints ===" -ForegroundColor Cyan
$bce = (Get-ChildItem "$OUT\baseline_seed42"   -Recurse -Filter *.pt -ErrorAction SilentlyContinue).Count
$mce = (Get-ChildItem "$OUT\morphaware_seed42" -Recurse -Filter *.pt -ErrorAction SilentlyContinue).Count
Write-Host "  baseline checkpoints:   $bce"
Write-Host "  morphaware checkpoints: $mce"
if ($bce -ge 1 -and $mce -ge 1) {
    Write-Host "SMOKE TEST PASSED — pipeline runs end-to-end. Safe to launch on the 5090." -ForegroundColor Green
} else {
    throw "SMOKE TEST FAILED — no checkpoint written"
}
