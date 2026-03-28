# ==========================================================
# Dockerfile — Sorani Kurdish GEC
# ==========================================================
# Minimal reproducible environment for training and inference.
#
# ARCH-2: The web module lives at Implementation/web/ (sibling dir).
# To include it, build from the Implementation/ directory:
#
#   docker build -t sorani-gec -f sorani-gec/Dockerfile .
#
# Run inference:
#   docker run --gpus all sorani-gec python scripts/10_infer.py \
#       --model results/models/morphaware/best_model.pt --morphaware \
#       --text "ئەو کتێبەکان خوێندمەوە"
#
# Run training:
#   docker run --gpus all -v $(pwd)/data:/app/data -v $(pwd)/results:/app/results \
#       sorani-gec python scripts/05_train_baseline.py --fp16
#
# Launch web UI:
#   docker run --gpus all -p 7860:7860 sorani-gec python -m web.app --share
# ==========================================================

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    LANG=C.UTF-8

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3-pip \
    libhunspell-dev hunspell git \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python

WORKDIR /app

# Install Python dependencies first (cache layer)
COPY sorani-gec/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source (build context = Implementation/)
COPY sorani-gec/pyproject.toml .
COPY sorani-gec/src/ src/
COPY sorani-gec/scripts/ scripts/
COPY sorani-gec/configs/ configs/
COPY sorani-gec/data/ data/
COPY sorani-gec/results/ results/

# ARCH-2 fix: Copy web module from sibling directory
COPY web/ web/

# Default: show help
CMD ["python", "scripts/10_infer.py", "--help"]
