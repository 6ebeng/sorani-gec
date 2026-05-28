#!/usr/bin/env bash
# remote_setup.sh — run on vast.ai GPU instance to install deps and launch Phase D
set -e

REPO=/workspace/sorani-gec
LOG=$REPO/results/phase_d/setup.log

mkdir -p $REPO/results/phase_d
exec > >(tee -a $LOG) 2>&1

echo "=== $(date) === Starting remote setup ==="

cd $REPO

# 1. Install PyTorch (cu128 for CUDA 13.0)
echo "--- Installing PyTorch ---"
pip3 install --quiet torch --index-url https://download.pytorch.org/whl/cu128
python3 -c "import torch; print('PyTorch', torch.__version__, '|', torch.cuda.device_count(), 'GPUs |', torch.cuda.get_device_name(0))"

# 2. Install project dependencies (pyhunspell not importable on Py 3.12 / not used at runtime)
echo "--- Installing project deps ---"
grep -v pyhunspell requirements.txt | pip3 install --quiet -r /dev/stdin
pip3 install --quiet sentencepiece sacremoses  # needed by back-translation script

# 3. Install project in editable mode
echo "--- Installing project package ---"
pip3 install --quiet -e .

# 4. Quick sanity checks
echo "--- Checking transforms + sentencepiece ---"
python3 -c "from transformers import T5Tokenizer; print('transformers OK')"
python3 -c "import sentencepiece; print('sentencepiece OK')"

# 5. Verify data
echo "--- Verifying splits_v2 ---"
wc -l data/splits_v2/*.jsonl

# 6. Run config CI test
echo "--- Config consistency tests ---"
python3 -m pytest tests/test_config_consistency.py -v --tb=short

echo "=== $(date) === Setup complete — launching Phase D training ==="

# 7. Launch training (all 3 seeds × 2 models)
bash scripts/run_phase_d_seeds.sh \
  2>&1 | tee results/phase_d/phase_d_train.log

echo "=== $(date) === All training runs complete ==="
