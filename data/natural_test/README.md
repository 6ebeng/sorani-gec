# Natural Test Set — Organic Sorani Error Collection

This directory holds **human-generated** Sorani Kurdish sentences with naturally
occurring grammatical errors, collected to complement the synthetic pipeline's
training and dev data. The goal is a minimum of **500 sentences** with organic
errors for final test-set evaluation.

## Files

- `sentences.jsonl` — one JSON object per collected sentence.
- `annotations.m2` — gold M² file with span-level annotations (phase 1.4).
- `sources.md` — provenance of each sentence cluster (where it came from,
  consent, licensing).

## Schema (`sentences.jsonl`)

```json
{
    "id": "nat_000001",
    "source_text": "original Sorani sentence as it appeared (with errors)",
    "target_text": "corrected reference produced by annotator(s)",
    "source_url": "https://... or 'learner_submission' / 'social_media' etc.",
    "register": "news | social | blog | learner | chat | forum",
    "dialect": "central | northern | southern | mixed | unknown",
    "error_types": ["subject_verb", "orthography", ...],
    "annotator_ids": ["A1", "A2"],
    "notes": "free-form; e.g. dialectal variant, ambiguous reading"
}
```

## Collection plan (phase 1.3)

Target domains (mix to avoid register bias):

1. **Learner corpora** — essays / short written texts from L2 Sorani learners
   (ideally via UKH linguistics coursemates, with verbal consent logged).
2. **Social media** — Twitter/X, Facebook public pages. Keep source URLs;
   drop anything private.
3. **Forum posts** — Kurdish-language forums (bwar.krd, kurdipedia comments).
4. **Machine-translated Kurdish** that native speakers reject as ungrammatical.

Per-item checklist:

- [ ] Source recorded (URL or consent receipt).
- [ ] Sentence transcribed verbatim (preserve original errors).
- [ ] Normalisation pass: Arabic → Sorani code points only, no content edits.
- [ ] Reference correction by annotator A1.
- [ ] Independent review by annotator A2; disagreements logged.
- [ ] Error type(s) tagged using the 25-class taxonomy from Ch. 6.

## Phase 1.4 — M² output

Once `sentences.jsonl` reaches ~100 items, run:

```pwsh
python scripts/build_m2_from_jsonl.py `
    --input data/natural_test/sentences.jsonl `
    --output data/natural_test/annotations.m2
```

This emits ERRANT-compatible M² using the character-span aligner in
`src/evaluation/m2_scorer.py`.

## Status

- 0 / 500 sentences collected (organic).
- **Review CSV seeded** at `sentences_for_review.csv` — 287 rows drawn from
  the active edited subset of `data/splits/test.jsonl`. These are
  synthetic-derived placeholders showing the schema with real Kurdish
  text; treat each row as a candidate to keep, edit into a real organic
  example, or mark `keep=n` to drop. Open in Excel (UTF-8 BOM written so
  Kurdish renders) or in VS Code.
- Round-trip:
  ```pwsh
  python scripts/export_natural_csv.py        # regenerate review CSV
  python scripts/csv_to_natural_jsonl.py      # CSV -> sentences.jsonl
  python scripts/build_m2_from_jsonl.py `
      --input  data/natural_test/sentences.jsonl `
      --output data/natural_test/annotations.m2
  ```
- Collection script scaffold: `scripts/collect_natural_errors.py`.
- M² build script: `scripts/build_m2_from_jsonl.py` (accepts the JSONL above
  _and_ the `source/target/errors` schema used in `data/splits/`; trivial
  copy-pairs are skipped).
- Consent log: pending (awaiting IRB-equivalent clearance from UKH
  Department of CSE).
