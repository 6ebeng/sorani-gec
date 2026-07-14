# Morphological Analysis

The morphology layer turns a raw Sorani sentence into per-word feature vectors and an agreement graph. It is rule-based (no trained tagger exists for Sorani at the quality needed), grounded in the grammar-book findings F#1–F#378.

## Modules (`src/morphology/`)

| Module                              | Role                                                                                                                                |
| ----------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `constants.py`                      | Linguistic constants: clitic inventories, verb affixes, case roles, demonstratives — the codified findings                          |
| `analyzer.py`                       | `MorphologicalAnalyzer` — segment + analyze each word (KLPT is an optional dependency; the analyzer degrades gracefully without it) |
| `features.py`                       | `FeatureExtractor` — 9 features per word                                                                                            |
| `graph.py`                          | `AgreementEdge`, `AgreementGraph` — 33 typed edge kinds (`EDGE_TYPE_ORDER`)                                                         |
| `builder.py`                        | Builds the graph from an analyzed sentence                                                                                          |
| `lexicon.py`                        | `SoraniLexicon` — Hunspell-backed lookup (33,856 entries, 6,387 affix rules from `data/hunspell/ckb-Arab.{dic,aff}`)                |
| `agreement.py`, `lexicon_parser.py` | Thin backward-compatibility shims re-exporting the above                                                                            |

Import order is acyclic: `constants → analyzer → graph → builder → agreement`.

## The 9 features

Per word, the extractor emits categorical values for:

1. `person` (1/2/3)
2. `number` (sg/pl)
3. `tense` (past/present/…)
4. `aspect`
5. `case`
6. `definiteness`
7. `transitivity`
8. `clitic_person`
9. `clitic_number`

Each becomes an embedding index in the morphology-aware model (see [[Model-Architecture]]).

## The agreement graph

Sorani split ergativity: in past transitive clauses, agent marking appears as a pronominal clitic (often displaced onto an object or preverbal element) while the verb agrees with the patient; in present tense the verb agrees with the subject via affixes. The graph makes those long-distance dependencies explicit: nodes are words, and 33 typed directed edges connect controllers to targets (subject→verb, possessor→clitic, noun→adjective, quantifier→noun, cross-clause links, …).

`build_agreement_graph(sentence)` returns the graph used two ways:

- as input structure for the model's agreement-prediction head,
- as the basis of the 14-check agreement accuracy metric ([[Evaluation-Metrics]]).

## Lexicon

`SoraniLexicon` searches `data/hunspell/ckb-Arab.dic` first, then `data/lexicon/`. It answers stem lookups, affix expansion, and suggestion queries; the spell checker (`src/data/spell_checker.py`) and several error generators depend on it. Download via `scripts/01a_download_ahmadi_lexicon.py`.

Next: [[Model-Architecture]]
