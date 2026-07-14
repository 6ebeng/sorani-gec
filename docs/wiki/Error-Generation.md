# Error Generation

Synthetic training pairs come from injecting linguistically-motivated errors into clean sentences. Every generator subclasses `BaseErrorGenerator` (`src/errors/base.py`) and returns character-offset annotations, so each pair carries exact edit spans.

## The 25 generators (`src/errors/`)

### Agreement errors (the thesis focus)

| Generator | Error injected |
|---|---|
| `subject_verb.py` | Subject–verb number disagreement (e.g., singular subject + plural verb) |
| `tense_agreement.py` | Tense/agreement marking on the wrong element under split ergativity |
| `clitic.py` | Wrong pronominal clitic form (person/number) |
| `possessive_clitic.py` | Possessive clitic mismatch |
| `noun_adjective.py` | Noun–adjective ezafe mismatch |
| `quantifier_agreement.py` | Quantifier–noun number clash |
| `conditional_agreement.py` | Agreement inside conditional clauses |
| `cross_clause_agreement.py` | Agreement across clause boundaries |
| `negative_concord.py` | Negative concord violations |
| `relative_clause.py` | Relative-clause agreement errors |
| `demonstrative_contraction.py` | Demonstrative contraction misuse |

### Morphosyntax and word order

| Generator | Error injected |
|---|---|
| `word_order.py` | SOV violations |
| `morpheme_order.py` | Morpheme sequencing inside the verb complex |
| `syntax_roles.py` | Role-marking confusion |
| `participle_swap.py` | Participle form swaps |
| `adverb_verb_tense.py` | Adverb–verb tense clash |
| `vocative_imperative.py` | Vocative/imperative form errors |
| `polite_imperative.py` | Politeness-register imperative errors |
| `preposition_fusion.py` | Fused-preposition errors |
| `adversative.py` | Adversative conjunction misuse |
| `dialectal.py` | Dialectal (non-standard) substitutions |

### Surface noise

| Generator | Error injected |
|---|---|
| `orthography.py` | Script/orthographic errors |
| `spelling_confusion.py` | Morphophonemically-confusable character swaps |
| `whitespace_error.py` | Whitespace insertion/deletion (ZWNJ-sensitive) |
| `punctuation_error.py` | Punctuation errors |

## Pipeline (`src/errors/pipeline.py`)

`03_generate_errors.py` drives the pipeline: for each clean sentence it samples generators by configured weights, applies at most one edit per pair (**single-edit contract** in the released splits, enforced by `create_splits_v2.py`), and emits JSONL records:

```json
{
  "original": "…clean…",
  "corrupted": "…with error…",
  "source": "…model input…",
  "target": "…model output…",
  "errors": [{"type": "subject_verb", "start": 12, "end": 18, "description": "…"}]
}
```

Grammatical knowledge behind the generators is grounded in 375 documented findings (F#1–F#378) extracted from 12 primary Kurdish grammar sources; the constants live in `src/morphology/constants.py`.

Test-set error distribution (397 edited pairs): noun–adjective (43), subject–verb (25), clitic form (24), possessive clitic (23) lead; the four surface-noise categories together are 29.0%.

Next: [[Morphological-Analysis]]
