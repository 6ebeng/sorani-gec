# Sorani Kurdish GEC — Implementation

## Agreement-Aware Grammatical Error Correction for Sorani Kurdish

A morphology-driven neural approach to grammatical error correction (GEC) for Sorani (Central) Kurdish, focusing on agreement errors.

### Project Structure

```
sorani-gec/
├── README.md
├── requirements.txt
├── pyproject.toml               # Package metadata (pip install -e .)
├── Dockerfile                   # NVIDIA CUDA container for GPU inference
├── configs/
│   └── default.yaml             # Training/eval configuration
├── data/
│   ├── raw/                     # Original Sorani text sources
│   ├── clean/                   # Normalized, deduplicated, sentence-split
│   ├── synthetic/               # Generated noisy→clean pairs
│   ├── splits/                  # Train/dev/test splits
│   └── hunspell/                # KurdishHunspell dictionary (33,856 entries)
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── collector.py         # Wikipedia API + local file corpus collector
│   │   ├── normalizer.py        # Arabic-script normalization & sentence splitting
│   │   ├── sorani_detector.py   # Sorani vs non-Sorani language detection
│   │   ├── sanitizer.py         # Post-collection corpus sanitization
│   │   ├── spell_checker.py     # Pyhunspell-based Sorani spell checker
│   │   ├── augmentation.py      # Data augmentation (synonym, swap, delete)
│   │   ├── curriculum.py        # Epoch-aware difficulty-ramped sampling
│   │   ├── tokenize.py          # Sorani-aware tokenization utility
│   │   └── splitter.py          # Stratified train/dev/test splitting
│   ├── errors/                  # 25 error generators (ABC pattern)
│   │   ├── __init__.py
│   │   ├── base.py              # BaseErrorGenerator ABC + ErrorResult
│   │   ├── subject_verb.py      # Subject-verb number disagreement
│   │   ├── noun_adjective.py    # Noun-adjective ezafe mismatch
│   │   ├── clitic.py            # Incorrect pronominal clitic forms
│   │   ├── tense_agreement.py   # Tense-agreement (split-ergative)
│   │   ├── possessive_clitic.py # Possessive clitic errors
│   │   ├── conditional_agreement.py  # Conditional clause agreement
│   │   ├── quantifier_agreement.py   # Quantifier-noun number
│   │   ├── demonstrative_contraction.py  # Demonstrative errors
│   │   ├── syntax_roles.py      # Case role / preposition errors
│   │   ├── dialectal.py         # Dialectal participle interchange
│   │   ├── relative_clause.py   # Relative clause agreement
│   │   ├── adversative.py       # Adversative connector errors
│   │   ├── participle_swap.py   # Agent↔patient participle swap
│   │   ├── orthography.py       # Orthographic/script errors (22 patterns)
│   │   ├── negative_concord.py  # Negation concord violations
│   │   ├── vocative_imperative.py    # Vocative/imperative errors
│   │   ├── adverb_verb_tense.py # Adverb-verb tense mismatch
│   │   ├── preposition_fusion.py     # Preposition fusion errors
│   │   ├── polite_imperative.py # Polite imperative errors
│   │   ├── spelling_confusion.py     # Morphophonemic character confusion
│   │   ├── word_order.py        # SOV word order violations
│   │   ├── whitespace_error.py  # Word fusion/splitting errors
│   │   ├── punctuation_error.py # Arabic/Latin punctuation swap
│   │   ├── cross_clause_agreement.py # Cross-clause tense concordance
│   │   ├── morpheme_order.py    # Prefix/suffix reordering errors
│   │   └── pipeline.py          # Synthetic corpus generation pipeline
│   ├── morphology/
│   │   ├── __init__.py
│   │   ├── analyzer.py          # Morphological analyzer (12 POS tags)
│   │   ├── features.py          # 9 morphological features extraction
│   │   ├── builder.py           # Agreement graph construction (two-law system)
│   │   ├── graph.py             # AgreementGraph with 33 edge types
│   │   ├── constants.py         # Linguistic constants (F#1–F#256)
│   │   └── lexicon.py           # Sorani lexicon (.dic + .aff parser)
│   ├── model/
│   │   ├── __init__.py
│   │   ├── baseline.py          # BaselineGEC (ByT5-small, byte-level)
│   │   ├── morphology_aware.py  # MorphologyAwareGEC (ByT5 + morph + agr)
│   │   └── ensemble.py          # Majority-vote / best-score ensemble
│   └── evaluation/
│       ├── __init__.py
│       ├── f05_scorer.py        # F₀.₅ computation (LCS-based edits)
│       ├── agreement_accuracy.py # 14 Sorani agreement checks
│       ├── gleu_scorer.py       # GLEU metric with bootstrap CIs
│       ├── m2_scorer.py         # M² scorer for GEC evaluation
│       └── inter_rater.py       # Cohen's κ / Fleiss' κ inter-annotator
├── scripts/
│   ├── 01_collect_data.py       # Step 1: Collect raw Sorani text
│   ├── 01a_download_ahmadi_lexicon.py  # Step 1a: Download lexicon data
│   ├── 01b_sanitize.py          # Step 1b: Corpus sanitization
│   ├── 02_normalize.py          # Step 2: Normalize and clean
│   ├── 03_generate_errors.py    # Step 3: Generate synthetic errors
│   ├── 04_split_data.py         # Step 4: Stratified train/dev/test split
│   ├── 04a_corpus_statistics.py # Step 4a: Error distribution analysis
│   ├── 05_train_baseline.py     # Step 5: Train ByT5 baseline model
│   ├── 06_train_morphaware.py   # Step 6: Train morphology-aware model
│   ├── 07_evaluate.py           # Step 7: Evaluation (F₀.₅, GLEU, M²)
│   ├── 08_ablation.py           # Step 8: Ablation studies
│   ├── 09_export_onnx.py        # Step 9: ONNX export for deployment
│   ├── 10_infer.py              # Step 10: CLI inference
│   ├── 11_hash_data.py          # Step 11: SHA-256 data integrity
│   └── 12_hyperparam_search.py  # Step 12: Hyperparameter search
├── tests/                       # 574 tests across 25 files
│   ├── test_normalizer.py
│   ├── test_error_generators.py
│   ├── test_evaluation.py
│   ├── test_morphology.py
│   ├── test_model.py
│   ├── test_pipeline.py
│   ├── test_pipeline_integration.py
│   ├── test_integration.py
│   ├── test_collector.py
│   ├── test_sorani_detector.py
│   ├── test_splitter.py
│   ├── test_augmentation.py
│   ├── test_baseline_model.py
│   ├── test_curriculum.py
│   ├── test_ensemble.py
│   ├── test_gleu_scorer.py
│   ├── test_inter_rater.py
│   ├── test_lexicon.py
│   ├── test_m2_scorer.py
│   ├── test_spell_checker.py
│   ├── test_training_scripts.py
│   ├── test_sanitizer.py
│   ├── test_corpus_catalog.py
│   ├── test_hyperparam_search.py
│   └── test_tokenize.py
└── results/
    ├── models/                  # Saved model checkpoints
    ├── metrics/                 # Evaluation metrics
    └── figures/                 # Plots and visualizations
```

### Quick Start

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows

# Install as editable package (recommended)
pip install -e ".[dev,web,logging]"

# Or install from requirements.txt
pip install -r requirements.txt

# Run the full pipeline step by step
python scripts/01_collect_data.py
python scripts/01a_download_ahmadi_lexicon.py
python scripts/01b_sanitize.py
python scripts/02_normalize.py
python scripts/03_generate_errors.py
python scripts/04_split_data.py
python scripts/04a_corpus_statistics.py
python scripts/05_train_baseline.py
python scripts/06_train_morphaware.py
python scripts/07_evaluate.py
python scripts/08_ablation.py
python scripts/09_export_onnx.py
python scripts/10_infer.py              # CLI inference on a sentence
python scripts/11_hash_data.py           # SHA-256 data integrity check
python scripts/12_hyperparam_search.py   # Hyperparameter search

# Run tests (574 tests)
pytest tests/ -v

# Reproducibility shortcut (via Makefile)
make reproduce
```

### Docker

```bash
# Build and run with Docker Compose (GPU + web UI)
docker compose up --build

# Or build the standalone container
docker build -t sorani-gec .
docker run --gpus all -p 7860:7860 sorani-gec
```

### Web Interface

A Gradio-based web interface is provided in `../web/` (sibling directory). It includes:

- Interactive GEC correction with model selection (Baseline / Morphology-Aware / Ensemble)
- Morphological analysis tab with feature extraction and agreement graph
- Evaluation tab with F₀.₅, GLEU, agreement accuracy metrics
- Side-by-side comparison of model outputs with diff highlighting

### Research Objectives

1. Develop a synthetic error-annotated dataset (~50,000 sentence pairs) of correct/erroneous Sorani Kurdish
2. Design a morphology-aware neural GEC model using ByT5 (byte-level Transformer)
3. Evaluate using F₀.₅, agreement-accuracy, and human evaluation

### License

MIT License (see LICENSE file)
