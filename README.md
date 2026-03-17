# Sorani Kurdish GEC — Implementation

## Agreement-Aware Grammatical Error Correction for Sorani Kurdish

A morphology-driven neural approach to grammatical error correction (GEC) for Sorani (Central) Kurdish, focusing on agreement errors.

### Project Structure

```
sorani-gec/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── setup.py                     # Package setup
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
│   │   ├── collector.py         # Corpus collection utilities
│   │   ├── normalizer.py        # Arabic-script normalization
│   │   ├── tokenizer.py         # SentencePiece tokenization
│   │   └── splitter.py          # Train/dev/test splitting
│   ├── errors/
│   │   ├── __init__.py
│   │   ├── base.py              # Base error generator class
│   │   ├── subject_verb.py      # Subject-verb number disagreement
│   │   ├── noun_adjective.py    # Noun-adjective Ezafe mismatch
│   │   ├── clitic.py            # Incorrect pronominal clitic forms
│   │   ├── tense_agreement.py   # Tense-agreement (split-ergative)
│   │   └── pipeline.py          # Synthetic corpus generation pipeline
│   ├── morphology/
│   │   ├── __init__.py
│   │   ├── analyzer.py          # Kurdish-BLARK morphological analyzer wrapper
│   │   ├── features.py          # Feature extraction (person, number, case, etc.)
│   │   └── agreement.py         # Agreement graph construction
│   ├── model/
│   │   ├── __init__.py
│   │   ├── baseline.py          # Baseline Transformer (no morphology)
│   │   ├── morphology_aware.py  # Morphology-aware Transformer
│   │   ├── embeddings.py        # Token + morphological feature embeddings
│   │   └── trainer.py           # Training loop
│   └── evaluation/
│       ├── __init__.py
│       ├── f05_scorer.py        # F₀.₅ computation
│       ├── agreement_accuracy.py # Sorani agreement-accuracy checker
│       ├── errant_kurdish.py    # ERRANT-style error analysis for Kurdish
│       └── human_eval.py        # Human evaluation utilities
├── scripts/
│   ├── 01_collect_data.py       # Step 1: Collect raw Sorani text
│   ├── 02_normalize.py          # Step 2: Normalize and clean
│   ├── 03_generate_errors.py    # Step 3: Generate synthetic errors
│   ├── 04_train_tokenizer.py    # Step 4: Train SentencePiece model
│   ├── 05_train_baseline.py     # Step 5: Train baseline model
│   ├── 06_train_morphaware.py   # Step 6: Train morphology-aware model
│   ├── 07_evaluate.py           # Step 7: Run evaluation
│   └── 08_ablation.py           # Step 8: Ablation studies
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_error_analysis.ipynb
│   └── 03_results_visualization.ipynb
├── tests/
│   ├── test_normalizer.py
│   ├── test_error_generators.py
│   └── test_evaluation.py
└── results/
    ├── models/                  # Saved model checkpoints
    ├── metrics/                 # Evaluation metrics
    └── figures/                 # Plots and visualizations
```

### Quick Start

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run the pipeline step by step
python scripts/01_collect_data.py
python scripts/02_normalize.py
python scripts/03_generate_errors.py
python scripts/04_train_tokenizer.py
python scripts/05_train_baseline.py
python scripts/06_train_morphaware.py
python scripts/07_evaluate.py
python scripts/08_ablation.py
```

### Research Objectives

1. Develop a synthetic error-annotated dataset (~50,000 sentence pairs) of correct/erroneous Sorani Kurdish
2. Design a morphology-aware neural GEC model using Transformer encoder-decoder
3. Evaluate using F₀.₅, agreement-accuracy, and human evaluation

### License

TBD (to be discussed with supervisor regarding Kurdish-BLARK alignment)
