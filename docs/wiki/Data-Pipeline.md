# Data Pipeline

From raw Kurdish text to train/dev/test splits. All data directories are gitignored; every artifact regenerates from the numbered scripts.

## Sources

| Source | Script | Notes |
|---|---|---|
| OCR'd Kurdish dissertations & linguistics books | `01_collect_data.py`, `ingest_dissertations.py` | Proofread academic prose; OCR quality audited (`ocr_audit.py` → `results/ocr_audit/`) |
| Kurdish Textbooks Corpus (KTC) | `01a_download_ktc.py`, `ingest_ktc.py` | Per-subject categories mapped via `src/data/corpus_catalog.py` |
| Ahmadi Hunspell lexicon (ckb-Arab) | `01a_download_ahmadi_lexicon.py` | 33,856 entries + 6,387 affix rules; used by the spell checker and morphology lexicon (`data/hunspell/`) |
| Natural test sentences | `13_collect_natural_test_sentences.py` → `export_natural_csv.py` → manual review → `csv_to_natural_jsonl.py` | 200+ real-world sentences in `data/natural_test/` |

## Steps

```bash
python scripts/01_collect_data.py        # gather raw text → data/raw/
python scripts/01b_sanitize.py --sorani-detect
python scripts/01c_balance_corpus.py     # balance across academic categories → data/balanced/
python scripts/02_normalize.py           # Unicode normalization → data/clean/
python scripts/03_generate_errors.py     # synthetic pairs → data/synthetic/annotations.jsonl
python scripts/04_split_data.py          # stratified split
python scripts/04a_corpus_statistics.py  # distribution stats + figures
python scripts/11_hash_data.py           # SHA-256 manifest of all artifacts
```

Or `make collect sanitize normalize generate split stats`.

### Sanitizer (`src/data/sanitizer.py`)

Nine filter stages per line: URL and wiki-template removal, mojibake detection, length gates, digit-ratio gates, duplicate removal, and (optionally) the `SoraniDetector` language filter. Detector accuracy was validated on a multilingual gold set (`results/sorani_detector_validation.json`).

### Normalizer (`src/data/normalizer.py`)

Arabic-script Unicode normalization: Arabic Yeh/Kaf → Kurdish Yeh/Keheh, Heh variants, ZWNJ handling, digit unification, whitespace collapse. Normalization decisions came from corpus counts, not assumptions; several edge cases (e.g., ة in Arabic loanwords) are documented in the thesis.

### Category-label contamination (important)

The scaled corpus originally carried `category\tsentence` prefixes. An early campaign accidentally trained with those labels embedded in target sequences, which pinned F₀.₅ near zero. `15_clean_corpus.py` strips the labels; `data/synthetic_scaled_clean/` is the cleaned pool. The contaminated pool has been deleted. See [[Training-Campaigns]].

## Splits chronology

| Splits | Built by | Size (train/dev/test) | Role |
|---|---|---|---|
| `splits` (v1) | `04_split_data.py` | ~superseded | First cut; deleted locally |
| **`splits_v2`** | `create_splits_v2.py` (+ `augment_test_v2.py`) | 5,253 / 465 / 647 | **Canonical dev/test.** Single-edit filter, Jaccard-0.90 cross-split dedup, SHA-256 manifest. Eleven thin error types augmented to n=20 test pairs each |
| **`splits_scaled`** | `14_build_scaled_train.py` | 26,841 / 465 / 647 | **Clean-campaign training set.** Train pool scaled ~5–7×; dev/test byte-identical to splits_v2 |

Test-set composition: 647 pairs total, of which 397 contain an injected error (the "edited subset") and 250 are copy-through pairs. Headline metrics use all 647; per-type analysis uses the 397.

Data integrity: `scripts/11_hash_data.py` writes SHA-256 hashes for every artifact; `scripts/diagnose_data.py` audits leakage, trivial-pair ratio, and error distribution (`results/data_diagnosis/report.md`).

Next: [[Error-Generation]]
