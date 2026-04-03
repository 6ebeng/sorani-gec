# Sorani Kurdish GEC — Makefile
# ================================
# Reproducible test and build commands for the thesis project.
#
# Environment: Python 3.10+, virtual environment at ../../.venv
# Run from: Implementation/sorani-gec/

PYTHON ?= python
PYTEST ?= $(PYTHON) -m pytest
VENV_PYTHON ?= ../../.venv/Scripts/python.exe

# ---------- Testing ----------

.PHONY: test test-quick test-web test-all lint

test:
	$(VENV_PYTHON) -m pytest tests/ -q --tb=short

test-quick:
	$(VENV_PYTHON) -m pytest tests/ -q --tb=short -x

test-web:
	cd ../.. && $(VENV_PYTHON) -m pytest Implementation/web/tests -q --tb=short

test-all: test test-web

lint:
	$(VENV_PYTHON) -m flake8 src/ scripts/ --max-line-length 100 --count --show-source --statistics

# ---------- Pipeline ----------

.PHONY: collect sanitize normalize generate split stats train-baseline train-morphaware evaluate ablation reproduce

collect:
	$(VENV_PYTHON) scripts/01_collect_data.py

sanitize:
	$(VENV_PYTHON) scripts/01b_sanitize.py --sorani-detect

normalize:
	$(VENV_PYTHON) scripts/02_normalize.py

generate:
	$(VENV_PYTHON) scripts/03_generate_errors.py

split:
	$(VENV_PYTHON) scripts/04_split_data.py

stats:
	$(VENV_PYTHON) scripts/04a_corpus_statistics.py

train-baseline:
	$(VENV_PYTHON) scripts/05_train_baseline.py --config configs/default.yaml --fp16

train-morphaware:
	$(VENV_PYTHON) scripts/06_train_morphaware.py --config configs/default.yaml --fp16

evaluate:
	$(VENV_PYTHON) scripts/07_evaluate.py --config configs/default.yaml

ablation:
	$(VENV_PYTHON) scripts/08_ablation.py --config configs/default.yaml

# ARCH-5: Full reproducible pipeline — one command from raw data to final metrics
reproduce: collect sanitize normalize generate split stats train-baseline train-morphaware evaluate ablation

# ---------- Hyperparameter Search ----------

.PHONY: hpsearch

hpsearch:
	$(VENV_PYTHON) scripts/12_hyperparam_search.py --config configs/default.yaml --fp16

# ---------- Utility ----------

.PHONY: hash-data

hash-data:
	$(VENV_PYTHON) scripts/11_hash_data.py
