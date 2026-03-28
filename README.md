# Sorani Kurdish GEC — Implementation

## Agreement-Aware Grammatical Error Correction for Sorani Kurdish

A morphology-driven neural approach to grammatical error correction (GEC) for Sorani (Central) Kurdish, focusing on agreement errors.

### Project Structure

```
sorani-gec/
├── README.md
├── requirements.txt
├── pyproject.toml               # Package metadata (pip-installable)
├── configs/
│   └── default.yaml             # Training/eval configuration
├── data/
│   ├── raw/                     # Original Sorani text sources
│   ├── clean/                   # Normalized, deduplicated, sentence-split
│   ├── synthetic/               # Generated noisy→clean pairs
│   └── splits/                  # Train/dev/test splits
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── collector.py         # Wikipedia API + local file corpus collector
│   │   ├── normalizer.py        # Arabic-script normalization & sentence splitting
│   │   ├── sorani_detector.py   # Sorani vs non-Sorani language detection
│   │   ├── spell_checker.py     # Pyhunspell-based Sorani spell checker
│   │   ├── augmentation.py      # Data augmentation (synonym, swap, delete)
│   │   └── splitter.py          # Stratified train/dev/test splitting
│   ├── errors/                  # 19 error generators (ABC pattern)
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
│   │   ├── orthography.py       # Orthographic/script errors
│   │   ├── negative_concord.py  # Negation concord violations
│   │   ├── vocative_imperative.py    # Vocative/imperative errors
│   │   ├── adverb_verb_tense.py # Adverb-verb tense mismatch
│   │   ├── preposition_fusion.py     # Preposition fusion errors
│   │   ├── polite_imperative.py # Polite imperative errors
│   │   └── pipeline.py          # Synthetic corpus generation pipeline
│   ├── morphology/
│   │   ├── __init__.py
│   │   ├── analyzer.py          # Morphological analyzer (KLPT fallback)
│   │   ├── features.py          # 9 morphological features extraction
│   │   ├── agreement.py         # Agreement rule checking (5 checks)
│   │   ├── builder.py           # 11-step agreement graph builder
│   │   ├── graph.py             # AgreementGraph with 24 edge types
│   │   ├── constants.py         # Linguistic constants (F#1-F#256)
│   │   ├── lexicon.py           # Morphological lexicon (32K+ entries)
│   │   └── lexicon_parser.py    # Ahmadi lexicon parser (6K+ affix rules)
│   ├── model/
│   │   ├── __init__.py
│   │   ├── baseline.py          # BaselineGEC (ByT5-small, byte-level)
│   │   ├── morphology_aware.py  # MorphologyAwareGEC (ByT5 + morph embed + agr)
│   │   └── ensemble.py          # Model ensemble for inference
│   └── evaluation/
│       ├── __init__.py
│       ├── f05_scorer.py        # F₀.₅ computation (LCS-based edits)
│       ├── agreement_accuracy.py # 5 Sorani agreement checks
│       ├── m2_scorer.py         # M² scorer for GEC evaluation
│       └── inter_rater.py       # Cohen's κ / Fleiss' κ inter-annotator
├── scripts/
│   ├── 01_collect_data.py       # Step 1: Collect raw Sorani text
│   ├── 01a_download_ahmadi_lexicon.py  # Step 1a: Download lexicon data
│   ├── 02_normalize.py          # Step 2: Normalize and clean
│   ├── 03_generate_errors.py    # Step 3: Generate synthetic errors
│   ├── 04_split_data.py         # Step 4: Stratified train/dev/test split
│   ├── 05_train_baseline.py     # Step 5: Train ByT5 baseline model
│   ├── 06_train_morphaware.py   # Step 6: Train morphology-aware model
│   ├── 07_evaluate.py           # Step 7: Run evaluation (F₀.₅ + agreement)
│   ├── 08_ablation.py           # Step 8: Ablation studies
│   └── 09_export_onnx.py        # Step 9: ONNX export for deployment
├── tests/
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
│   └── test_splitter.py
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

# Install as package (editable)
pip install -e ".[dev,web,logging]"

# Or install from requirements.txt
pip install -r requirements.txt

# Run the pipeline step by step
python scripts/01_collect_data.py
python scripts/01a_download_ahmadi_lexicon.py
python scripts/02_normalize.py
python scripts/03_generate_errors.py
python scripts/04_split_data.py
python scripts/05_train_baseline.py
python scripts/06_train_morphaware.py
python scripts/07_evaluate.py
python scripts/08_ablation.py

# Run tests
pytest tests/ -v
```

### Research Objectives

1. Develop a synthetic error-annotated dataset (~50,000 sentence pairs) of correct/erroneous Sorani Kurdish
2. Design a morphology-aware neural GEC model using ByT5 (byte-level Transformer)
3. Evaluate using F₀.₅, agreement-accuracy, and human evaluation

### License

TBD (to be discussed with supervisor regarding Kurdish-BLARK alignment)
