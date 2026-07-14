# Testing

668 tests total: 626 in `sorani-gec/tests/` + 42 in `../web/tests/`. Everything runs on CPU; no GPU or network needed (model-download tests are mocked/skipped).

## Running

```bash
make test          # sorani-gec suite
make test-web      # web suite
make test-all      # both
make lint          # flake8, max-line-length 100

# direct
pytest tests/ -q --tb=short
pytest tests/test_error_generators.py -v      # one file
pytest tests/ -q -k "agreement"               # by keyword
```

On Windows, set `$env:PYTHONIOENCODING = "utf-8"` first — many tests assert on Arabic-script strings.

## Suite map (28 files)

| Area | Files |
|---|---|
| Data pipeline | `test_collector.py`, `test_sanitizer.py`, `test_normalizer.py`, `test_splitter.py`, `test_sorani_detector.py`, `test_spell_checker.py`, `test_tokenize.py`, `test_corpus_catalog.py`, `test_corpus_coverage.py`, `test_augmentation.py`, `test_curriculum.py` |
| Error generation | `test_error_generators.py`, `test_pipeline.py`, `test_pipeline_integration.py` |
| Morphology | `test_morphology.py`, `test_lexicon.py` |
| Models | `test_model.py`, `test_baseline_model.py`, `test_ensemble.py`, `test_training_scripts.py`, `test_hyperparam_search.py` |
| Evaluation | `test_evaluation.py`, `test_gleu_scorer.py`, `test_m2_scorer.py`, `test_inter_rater.py` |
| Infrastructure | `test_config_consistency.py`, `test_requirements_sync.py`, `test_integration.py` |

`test_requirements_sync.py` fails if `requirements.txt` and `pyproject.toml` drift apart. `test_config_consistency.py` pins the YAML config schema against script argparse expectations.

## Conventions

- Pure pytest (no unittest classes), fixtures over setup methods
- Kurdish test strings are real Sorani, not lorem ipsum — regressions in normalization show up as visible text changes
- Logging uses %-formatting everywhere (enforced; no f-strings in log calls)

## Smoke test

`scripts/smoke_test_pipeline.py` regenerates tiny fixtures and exercises collect → generate → split → train (1 epoch, tiny model) → evaluate end-to-end in ~5–10 CPU minutes.

Next: [[Reproducibility]]
