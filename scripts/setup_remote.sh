#!/usr/bin/env bash
# Remote environment setup for the RTX 5090 (Blackwell, sm_120) instance.
# Run ONCE after extracting the bundle. Idempotent-ish; safe to re-run.
set -euo pipefail

PROJ="${PROJ:-$HOME/sorani-gec}"
cd "$PROJ"

echo "==> Python: $(python3 --version)"
echo "==> Creating venv (.venv) ..."
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools

echo "==> Installing torch (CUDA 12.8 wheel for Blackwell sm_120) ..."
pip install --index-url https://download.pytorch.org/whl/cu128 torch

echo "==> Installing slim training/eval deps ..."
pip install -r requirements-gpu.txt

echo "==> Installing project (editable, no deps) ..."
pip install -e . --no-deps || echo "WARN: editable install failed; relying on PYTHONPATH"

echo "==> Verifying CUDA + device capability ..."
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))  # expect (12, 0)
PY

echo "==> Setup complete. Activate with: source $PROJ/.venv/bin/activate"
