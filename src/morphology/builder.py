"""
Agreement Graph Builder

Constructs agreement graphs for Sorani Kurdish sentences using the two-law
system from Slevanayi (2001):
  Law 1 — Subject-verb agreement (person + number)
  Law 2 — Object-verb agreement (ergative, past transitive only)

All linguistic constants are imported from constants.py; graph structures
from graph.py; morphological analysis from analyzer.py.
"""

import logging

from .analyzer import MorphologicalAnalyzer, MorphFeatures
from .constants import (
    ALL_VERB_PREFIXES,
    CLITIC_FORMS,
    CLAUSE_INITIAL_ONLY_CONJUNCTIONS,
    COLLECTIVE_NOUNS,
    COMMON_PROPER_NOUNS,
    COMPLEMENT_REQUIRING_VERBS,
    COORDINATION_CONJUNCTION,
    DEFINITE_MARKER_MIGRATION_DESCRIPTIVE,
    DEMONSTRATIVES,
    EXISTENTIAL_STEMS,
    HERGIZ_ADVERBS,
    HERGIZ_BANNED_TENSES,
    INTRANSITIVE_PAST_STEMS,
    INTERROGATIVE_PRONOUNS,
    INVARIANT_ADJECTIVES,
    INVARIANT_POSSESSIVES,
    INVARIANT_PRONOUNS,
    MASS_NOUNS,
    MEASURE_WORDS,
    NEGATION_CONJUNCTIONS,
    NOUN_MARKING_SUFFIXES,
    PAST_VERB_STEMS,
    PERSON_HIERARCHY,
    PRE_HEAD_DETERMINERS,
    PRESENT_VERB_PREFIXES,
    PREVERB_TRANSITIVITY_FLIPS,
    QUANTIFIER_FORMS,
    RECIPROCAL_PRONOUNS,
    RELATIVE_CLAUSE_MARKER,
    SORANI_SUBORDINATING_CONJUNCTIONS,
    SUBJECT_PRONOUNS,
    TRANSITIVE_PAST_STEMS,
    YI_DOUBLE_SCENARIOS,
)
from .graph import AgreementEdge, AgreementGraph  # noqa: F401

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _token_starts_with_stem(token: str, stem: str) -> bool:
    """Check if token starts with a verb stem (boundary-aware).

    Avoids substring false positives: 'کر' should match 'کردم' but
    not 'ڕابەکردن'. Strips known prefixes before comparison.
    """
    prefixes = ("دە", "ئە", "نا", "نە", "ب", "مە",
                "وەر", "هەڵ", "لێ", "تێ", "دەر", "پێ")
    remaining = token
    for pfx in sorted(prefixes, key=len, reverse=True):
        if remaining.startswith(pfx) and len(remaining) > len(pfx):
            remaining = remaining[len(pfx):]
            break
    return remaining.startswith(stem) or token.startswith(stem)


def _is_present_verb(token: str) -> bool:
    """Check if a token looks like a present-tense verb (has دە-/ئە- prefix)."""
    return any(token.startswith(p) for p in PRESENT_VERB_PREFIXES)


def _is_past_verb(token: str, features: MorphFeatures) -> bool:
    """Check if a token looks like a past-tense verb.

    Past verbs lack present prefixes and contain a known past stem.
    Source: Amin (2016), pp. 51-52.
    """
    if _is_present_verb(token):
        return False
    if features.tense == "past":
        return True
    for stem in PAST_VERB_STEMS:
        if _token_starts_with_stem(token, stem):
            return True
    return False


def _is_transitive_past(token: str, features: MorphFeatures | None = None) -> bool:
    """Determine if a past verb is transitive (ergative agreement).

    Source: Slevanayi (2001), pp. 60-61 — transitive past verbs trigger
    Law 2 (object-verb agreement).
    """
    if features is not None and features.raw_analysis.get("ahmadi_transitive"):
        return True
    
    for stem in TRANSITIVE_PAST_STEMS:
        if _token_starts_with_stem(token, stem):
            return True
    return False


def _is_intransitive_past(token: str, features: MorphFeatures | None = None) -> bool:
    """Determine if a past verb is intransitive (subject agreement).
    """
    if features is not None and features.raw_analysis.get("ahmadi_intransitive"):
        return True
        
    for stem in INTRANSITIVE_PAST_STEMS:
        if _token_starts_with_stem(token, stem):
            return True
    return False


def _is_invariant(token: str) -> bool:
    """Check if a token is an invariant form that never agrees.

    Source: Slevanayi (2001), pp. 38-48, 77-78, 81, 82-83 — adjectives,
    reflexive pronouns, interrogative pronouns, reciprocal pronouns,
    and possessive pronouns are invariant.

    Note on formal coincidence (لێکچوونی شێوەیی, pp. 38-39, 83):
    When an invariant form happens to share features with the verb,
    this is coincidence, not true agreement. True agreement requires
    a controller→target relationship with feature copying.
    """
    return (
        token in INVARIANT_ADJECTIVES
        or token in INVARIANT_PRONOUNS
        or token in INVARIANT_POSSESSIVES
        or token in RECIPROCAL_PRONOUNS
    )


def _is_interrogative(token: str) -> bool:
    """Check if a token is an interrogative pronoun.

    Interrogative pronouns are completely invariant at every syntactic
    level — as subject, object, or indirect object. They NEVER trigger
    agreement with the verb.
    Source: Slevanayi (2001), p. 81.
    """
    return token in INTERROGATIVE_PRONOUNS


def _is_reciprocal(token: str) -> bool:
    """Check if a token is a reciprocal pronoun.

    Reciprocal pronouns (ئێکدوو, هەڤدوو, یەکتر) are always object
    (direct or indirect) and carry invariant form. When reciprocal is
    object of past transitive, the verb's features reflect the subject,
    not the reciprocal — any match is formal coincidence.
    Source: Slevanayi (2001), pp. 82-83.
    """
    return token in RECIPROCAL_PRONOUNS


def _is_quantifier(token: str) -> bool:
    """Check if a token is a quantifier/numeral.

    Quantifiers always trigger plural verb agreement but do NOT agree
    in number with the head noun (the noun stays singular after numerals).
    Source: Slevanayi (2001), pp. 87-88; Maaruf (2010), p. 139

    Important positional constraint from Mukriani (2000, pp. 24-26):
    Only PRE-NOMINAL quantifiers control verb agreement. Post-nominal
    ordinals (کەسی دووەم) do not trigger plural agreement.
    The caller (build_agreement_graph) should verify position before
    creating a quantifier_verb edge.
    """
    return token in QUANTIFIER_FORMS


def _is_mass_noun(token: str) -> bool:
    """Check if a token is a mass noun (never takes plural directly).

    Mass nouns require a measure word for quantification, and the measure
    word — not the mass noun — controls verb agreement.
    Source: Slevanayi (2001), pp. 46-47, 53, 57.
    """
    return token in MASS_NOUNS


def _is_measure_word(token: str) -> bool:
    """Check if a token is a measure word (پێوەر).

    Measure words are the agreement controller when quantifying mass nouns.
    Source: Slevanayi (2001), pp. 47, 53, 57.
    """
    return token in MEASURE_WORDS


def _is_collective_noun(token: str) -> bool:
    """Check if a token is a collective noun.

    Collective nouns are morphologically singular but semantically plural.
    Bare (no determiner) → singular verb; with هەموو → plural verb.
    Source: Slevanayi (2001), pp. 45-46, 58.
    """
    return token in COLLECTIVE_NOUNS


def _is_demonstrative(token: str) -> bool:
    """Check if a token is a demonstrative pronoun.

    Demonstratives have dual behavior (Slevanayi 2001, pp. 83-86):
      - As pro-form (standalone): agrees directly with verb
      - As determiner (in NP): agrees with head noun only
    """
    return token in DEMONSTRATIVES


def _is_proper_noun(token: str) -> bool:
    """Check if a token is a proper noun.

    Proper nouns cannot take indefinite (ەک) or plural (ان/ین) markers.
    They always refer to a unique entity.
    Source: Slevanayi (2001), pp. 43-44.
    """
    return token in COMMON_PROPER_NOUNS


def _is_bare_noun(token: str, features: MorphFeatures) -> bool:
    """Check if a token is a bare noun (no determiner, no plural marker).

    Bare nouns agree only in PERSON (3rd) with the verb, not number,
    because number is unmarked. The verb defaults to 3rd person singular.
    Source: Slevanayi (2001), pp. 55-56.
    """
    if features.pos != "NOUN":
        return False
    # Not bare if it carries any definiteness or plural marker
    for suffix in NOUN_MARKING_SUFFIXES:
        if token.endswith(suffix):
            return False
    # Not a bare noun if it's a pronoun, quantifier, or demonstrative
    if (token in SUBJECT_PRONOUNS
            or _is_quantifier(token)
            or _is_demonstrative(token)
            or _is_proper_noun(token)):
        return False
    return True


# Existential forms that are standalone words (not productively prefixable
# stems).  These must match the token exactly; startswith matching would
# cause false positives—e.g. "هەن" matching "هەندێ" (quantifier 'some'),
# "هەنار" (pomegranate), or "هەنگاو" (step).
_EXISTENTIAL_EXACT_FORMS = frozenset({"هەیە", "نییە", "هەبێت", "نەبێت", "هەن"})


def _is_existential_verb(token: str) -> bool:
    """Check if a token is an existential verb form (هەبوون).

    Existential هەبوون has three uses with different agreement:
      1. Becoming (بوونی هاتنەئارایی) — regular Law 1
      2. Existence — agreement set pronoun agrees
      3. Possession — possessive set NEVER agrees, verb stays 3sg
    Source: Slevanayi (2001), pp. 75-77.
    """
    if token in _EXISTENTIAL_EXACT_FORMS:
        return True
    for stem in EXISTENTIAL_STEMS:
        if stem in _EXISTENTIAL_EXACT_FORMS:
            continue  # already handled via exact match above
        if _token_starts_with_stem(token, stem):
            return True
    return False


def _is_vocative(token: str, features: MorphFeatures) -> bool:
    """Check if a token carries a vocative suffix.

    Vocative nouns agree with imperative verbs in number:
      کوڕۆ + وەرە (sg)  vs.  کوڕینۆ + وەرن (pl)
    Source: Slevanayi (2001), pp. 16, 72-73.

    Guards against false positives: verbs (دەچێ ends in ێ),
    loanwords (ڕادیۆ ends in ۆ), and other POS-tagged tokens
    that happen to share the same final character.
    """
    # Vocative is a nominal phenomenon — exclude verbs and
    # tokens the analyzer already tagged as non-nominal.
    if features.pos in ("VERB", "ADP", "ADV", "CONJ", "SCONJ", "PUNCT"):
        return False
    return (
        token.endswith("ۆ")
        or token.endswith("ێ")
        or token.endswith("ینۆ")
    )


def _is_clause_boundary(token: str, features: MorphFeatures) -> bool:
    """Check if a token marks a clause boundary.

    Clause boundaries prevent agreement edges from crossing between
    independent clauses. Markers include subordinating conjunctions,
    the relative clause marker کە, coordinating conjunctions that
    separate full clauses, punctuation, and infinitival forms.

    Source: Ibrahim (1988), pp. 75-142 — conjunction inventory;
    Slevanayi (2001) — clause segmentation for agreement.
    """
    if features.pos == "PUNCT":
        return True
    if features.pos == "SCONJ":
        return True
    if token == RELATIVE_CLAUSE_MARKER:
        return True
    # و between verbs often indicates clause boundary
    if token == COORDINATION_CONJUNCTION and features.pos == "CCONJ":
        return True
    # Subordinating conjunctions from constants (Ibrahim 1988)
    if token in SORANI_SUBORDINATING_CONJUNCTIONS:
        return True
    # Infinitival forms start a new clause boundary
    if getattr(features, "tense", "") == "infinitive":
        return True
    return False


def _has_clause_boundary_between(
    tokens: list[str],
    feat_list: list[MorphFeatures],
    start: int,
    end: int,
) -> bool:
    """Return True if any token strictly between start and end is a clause boundary."""
    lo, hi = min(start, end), max(start, end)
    for k in range(lo + 1, hi):
        if _is_clause_boundary(tokens[k], feat_list[k]):
            return True
    return False


def _resolve_compound_subject_person(person_numbers: list[tuple[str, str]]) -> tuple[str, str]:
    """Resolve agreement controller for compound (coordinated) subjects.

    When subjects of different persons are conjoined, the most familiar
    person controls agreement and the number becomes plural [F#75, F#87]:
      1st > 2nd > 3rd  (Slevanayi 2001, p. 68)

    Compound noun subjects always force plural [F#88] (Slevanayi 2001, p. 61).

    Examples:
      "من و تۆ" (I and you)     → 1pl (1st person controls)
      "تۆ و ئەو" (you and he)   → 2pl (2nd person controls)
    """
    if not person_numbers:
        return ("3", "sg")

    # Compound subject is always plural
    best_person = "3"
    for person, _ in person_numbers:
        if PERSON_HIERARCHY.get(person, 2) < PERSON_HIERARCHY.get(best_person, 2):
            best_person = person

    return (best_person, "pl")


def build_agreement_graph(
    sentence: str,
    analyzer: MorphologicalAnalyzer,
) -> AgreementGraph:
    """Build agreement graph for a Sorani Kurdish sentence.

    Implements the two-law agreement system from Slevanayi (2001) [F#66]:

    Law 1 — Subject-verb agreement (person + number) [F#1, F#94]:
      Applies when the verb is (a) intransitive in any tense, or
      (b) transitive in present/future tense.

    Law 2 — Object-verb agreement (person + number, ergative) [F#14, F#80]:
      Applies when the verb is transitive AND in past tense.
      The verb agrees with the object, not the subject.
      Source: Slevanayi (2001), pp. 60-61.

    Additional edge types:
      - NP-internal [F#67, F#90]: determiner → noun (number + definiteness),
        but NOT adjective → noun (adjectives are invariant per
        Slevanayi 2001, pp. 38-48 [F#79]). No definiteness agreement
        exists within the NP (Slevanayi 2001, pp. 47-48).
      - Clitic-referent [F#9, F#22]: clitic-bearing word → argument
        (person + number). Clitic roles swap by tense [F#22].
      - Compound subject resolution [F#75, F#87, F#88]: coordinated
        pronouns/nouns → verb, using the pronoun familiarity hierarchy
        (1st > 2nd > 3rd). Compound noun subjects always force plural
        (Slevanayi 2001, p. 61).
      - Interrogative/reciprocal subjects [F#73, F#74]: NO agreement edge
        built (Slevanayi 2001, pp. 81, 82-83).
      - Vocative-imperative [F#76]: vocative noun agrees with imperative
        verb in number.
      - Existential three-way [F#72]: بوون/هەبوون classified by use.
    """
    tokens = analyzer.tokenize(sentence)
    features = [analyzer.analyze_token(tok) for tok in tokens]

    graph = AgreementGraph(tokens=tokens, features=features)

    # ------------------------------------------------------------------
    # Step 1: Identify verbs and classify by tense + transitivity
    # ------------------------------------------------------------------
    # Prefer analyzer-assigned features (tense, transitivity) when
    # available; fall back to heuristic helpers only for gaps.
    verb_info: dict[int, dict] = {}  # idx → {tense, transitive, law}
    for i, tok in enumerate(tokens):
        is_verb = features[i].pos == "VERB"
        # Only use prefix-based and stem-based detection when the
        # analyzer hasn't already assigned a non-verb POS.  Words like
        # بچووک (ADJ) start with ب and contain past stem چوو, but
        # are not verbs.
        pos_allows_verb = features[i].pos in ("", "VERB")
        has_prefix = (
            any(tok.startswith(p) for p in ALL_VERB_PREFIXES)
            and pos_allows_verb
        )

        if is_verb or has_prefix or (pos_allows_verb and _is_past_verb(tok, features[i])):
            is_wistin = features[i].raw_analysis.get("is_wistin_exception", False)
            
            # Resolve tense: analyzer first, then heuristic
            if features[i].tense:
                tense = features[i].tense
            elif _is_present_verb(tok) or has_prefix:
                tense = "present"
            else:
                tense = "past"
            
            # Resolve transitivity: analyzer first, then heuristic
            if features[i].transitivity == "trans":
                is_trans = True
                is_intrans = False
            elif features[i].transitivity == "intrans":
                is_trans = False
                is_intrans = True
            else:
                is_trans = _is_transitive_past(tok, features[i])
                is_intrans = _is_intransitive_past(tok, features[i])
            
            if is_wistin:
                verb_info[i] = {
                    "tense": tense,
                    "transitive": True,
                    "law": "law1",
                    "wistin_exception": True
                }
            elif tense in ("present", "future", "imperative"):
                verb_info[i] = {
                    "tense": tense,
                    "transitive": None,
                    "law": "law1",
                }
            elif tense == "past":
                if is_trans:
                    verb_info[i] = {
                        "tense": "past",
                        "transitive": True,
                        "law": "law2",
                    }
                elif is_intrans:
                    verb_info[i] = {
                        "tense": "past",
                        "transitive": False,
                        "law": "law1",
                    }
                else:
                    verb_info[i] = {
                        "tense": "past",
                        "transitive": None,
                        "law": "law1",
                    }
            else:
                verb_info[i] = {
                    "tense": tense,
                    "transitive": None,
                    "law": "law1",
                }

            # Passive reframing: passive voice turns ergative (Law 2)
            # into nominative-accusative (Law 1) because the patient
            # is promoted to subject position.
            # Source: Slevanayi (2001), pp. 55-56 — passive demotes agent.
            if getattr(features[i], "voice", "") == "passive" and verb_info[i]["law"] == "law2":
                verb_info[i]["law"] = "law1"
                verb_info[i]["passive"] = True
                logger.debug(
                    "Passive reframe: '%s' at %d switched from Law 2 → Law 1",
                    tok, i,
                )

            # Compound verb transitivity flip (F#132):
            # Preverbs like هەڵ can reverse transitivity of the root.
            # E.g. کرد (transitive) → هەڵکرد (intransitive, "got up").
            # Source: Fatah & Qadir (2006), p. 42
            cpx = getattr(features[i], "compound_prefix", "")
            if cpx and cpx in PREVERB_TRANSITIVITY_FLIPS:
                flip_map = PREVERB_TRANSITIVITY_FLIPS[cpx]
                # Check if any root in the flip map matches the verb stem
                for root, new_trans in flip_map.items():
                    if root in tok:
                        old_law = verb_info[i]["law"]
                        if new_trans == "intransitive" and old_law == "law2":
                            verb_info[i]["law"] = "law1"
                            verb_info[i]["transitive"] = False
                            logger.debug(
                                "Transitivity flip: '%s' at %d — preverb %s "
                                "makes root %s intransitive (Law 2 → Law 1)",
                                tok, i, cpx, root,
                            )
                        elif new_trans == "transitive" and old_law == "law1" and tense == "past":
                            verb_info[i]["law"] = "law2"
                            verb_info[i]["transitive"] = True
                            logger.debug(
                                "Transitivity flip: '%s' at %d — preverb %s "
                                "makes root %s transitive (Law 1 → Law 2)",
                                tok, i, cpx, root,
                            )
                        break

    # M7 fix: Warn if any verb in verb_info has no person/number from analyzer.
    # Set 2 suffixes are extracted by the analyzer; if missing, edges will
    # lack the person/number features needed for agreement checking.
    for vi, vinfo in verb_info.items():
        if not features[vi].person:
            logger.warning(
                "Verb '%s' at %d has no person assigned by analyzer",
                tokens[vi], vi,
            )

    # ------------------------------------------------------------------
    # Step 2: Detect subject pronouns and compound subjects
    # ------------------------------------------------------------------
    # Look for coordinated pronouns: "من و تۆ", "ئەو و ئەوان", etc.
    subject_spans: list[dict] = []  # {indices, person, number}
    i = 0
    while i < len(tokens):
        if tokens[i] in SUBJECT_PRONOUNS:
            # Check for coordination: PRON و PRON
            coord_persons = [SUBJECT_PRONOUNS[tokens[i]]]
            coord_indices = [i]
            j = i + 1

            # Detect negation conjunction pattern: نە PRON و نە PRON
            # Source: Slevanayi (2001), p. 61 — first conjunct controls
            is_negation_conj = (
                i > 0 and tokens[i - 1] in NEGATION_CONJUNCTIONS
            )

            # Standard coordination: PRON و PRON
            while j + 1 < len(tokens) and tokens[j] == "و" and tokens[j + 1] in SUBJECT_PRONOUNS:
                coord_persons.append(SUBJECT_PRONOUNS[tokens[j + 1]])
                coord_indices.append(j + 1)
                j += 2

            # Also handle: نە PRON نە PRON (without و)
            while (j + 1 < len(tokens)
                   and tokens[j] in NEGATION_CONJUNCTIONS
                   and tokens[j + 1] in SUBJECT_PRONOUNS):
                coord_persons.append(SUBJECT_PRONOUNS[tokens[j + 1]])
                coord_indices.append(j + 1)
                is_negation_conj = True
                j += 2

            if len(coord_persons) > 1:
                if is_negation_conj:
                    # Negation conjunction: first conjunct controls
                    # agreement — verb matches FIRST subject only
                    # Source: Slevanayi (2001), p. 61
                    first = coord_persons[0]
                    subject_spans.append({
                        "indices": coord_indices,
                        "person": first[0],
                        "number": first[1],
                        "last_idx": coord_indices[-1],
                    })
                else:
                    # Regular compound subject — resolve via hierarchy
                    resolved = _resolve_compound_subject_person(coord_persons)
                    subject_spans.append({
                        "indices": coord_indices,
                        "person": resolved[0],
                        "number": resolved[1],
                        "last_idx": coord_indices[-1],
                    })
                i = j
            else:
                pn = SUBJECT_PRONOUNS[tokens[i]]
                subject_spans.append({
                    "indices": [i],
                    "person": pn[0],
                    "number": pn[1],
                    "last_idx": i,
                })
                i += 1
        else:
            i += 1

    # ------------------------------------------------------------------
    # Step 2a: Detect compound NOUN subjects (X و Y → plural verb)
    # ------------------------------------------------------------------
    # Source: Slevanayi (2001), p. 61 — "If NP is compound (two heads),
    # the verb is in plural form."
    # Scan for NOUN و NOUN patterns not already captured as pronouns.
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        # Skip if already captured in pronoun coordination
        already_captured = any(i in s["indices"] for s in subject_spans)
        if (not already_captured
                and features[i].pos == "NOUN"
                and not _is_interrogative(tok)
                and not _is_reciprocal(tok)):
            # Check for coordination: NOUN (ADJ)* و NOUN (ADJ)* ...
            # Skip adjective modifiers between head nouns and و,
            # since Sorani adjectives follow their head noun.
            coord_indices = [i]
            j = i + 1
            # Skip adjective modifiers after the head noun
            while j < len(tokens) and features[j].pos == "ADJ":
                j += 1
            while (j + 1 < len(tokens)
                   and tokens[j] == COORDINATION_CONJUNCTION
                   and features[j + 1].pos == "NOUN"):
                coord_indices.append(j + 1)
                j += 2
                # Skip adjective modifiers after the coordinated noun
                while j < len(tokens) and features[j].pos == "ADJ":
                    j += 1
            if len(coord_indices) > 1:
                # Compound noun subject → always 3rd person plural
                subject_spans.append({
                    "indices": coord_indices,
                    "person": "3",
                    "number": "pl",
                    "last_idx": coord_indices[-1],
                })
                i = j
            else:
                i += 1
        else:
            i += 1

    # ------------------------------------------------------------------
    # Step 2b: Filter out interrogative pronoun subjects
    # ------------------------------------------------------------------
    # Source: Slevanayi (2001), p. 81 — interrogative pronouns are
    # completely invariant. If one is detected as the sole subject,
    # do NOT build an agreement edge to the verb.
    subject_spans = [
        s for s in subject_spans
        if not any(_is_interrogative(tokens[idx]) for idx in s["indices"])
    ]

    # ------------------------------------------------------------------
    # Step 2c: Detect bare noun subjects (person-only agreement)
    # ------------------------------------------------------------------
    # Source: Slevanayi (2001), pp. 55-56
    # When a common noun appears bare (no det, no plural marker) as
    # subject, the verb agrees only in PERSON (3rd), not number.
    # In oblique case (object of past transitive), bare nouns agree
    # in NEITHER person NOR number (Slevanayi 2001, pp. 55-56).
    for i, tok in enumerate(tokens):
        if _is_bare_noun(tok, features[i]):
            # B2: Oblique-cased nouns are not subjects — they are
            # governed by a preposition and cannot control verb agreement.
            if getattr(features[i], "case", "") == "obl":
                continue
            # Check token is not already captured as pronoun subject
            already_subject = any(
                i in s["indices"] for s in subject_spans
            )
            if already_subject:
                continue
            # Check position: bare noun should precede a verb within
            # the same clause (no clause boundary between noun and verb)
            for vi in verb_info:
                if 0 < vi - i <= 8 and not _has_clause_boundary_between(tokens, features, i, vi):
                    subject_spans.append({
                        "indices": [i],
                        "person": "3",
                        "number": "",  # empty = person-only agreement
                        "last_idx": i,
                    })
                    break

    # ------------------------------------------------------------------
    # Step 2d: Detect definite/marked noun subjects
    # ------------------------------------------------------------------
    # Definite nouns (-eke, -ekan) and plural nouns (-an) are full
    # subjects that agree in person and number with the verb.
    # These are not bare nouns (Step 2c) and not pronouns (Step 2).
    # Source: Slevanayi (2001), pp. 55-56, 60-61.
    for i, tok in enumerate(tokens):
        if features[i].pos != "NOUN":
            continue
        if getattr(features[i], "case", "") == "obl":
            continue
        already_subject = any(i in s["indices"] for s in subject_spans)
        if already_subject:
            continue
        if _is_bare_noun(tok, features[i]):
            continue
        if _is_interrogative(tok) or _is_reciprocal(tok):
            continue
        if (_is_quantifier(tok) or _is_mass_noun(tok)
                or _is_collective_noun(tok) or _is_demonstrative(tok)
                or _is_proper_noun(tok)):
            continue
        # Marked noun (definite, plural, etc.) — check it precedes a verb
        for vi in verb_info:
            if 0 < vi - i <= 8 and not _has_clause_boundary_between(tokens, features, i, vi):
                noun_number = features[i].number or "sg"
                subject_spans.append({
                    "indices": [i],
                    "person": "3",
                    "number": noun_number,
                    "last_idx": i,
                })
                break

    # ------------------------------------------------------------------
    # Step 3: Link subjects/objects to verbs
    # ------------------------------------------------------------------
    # Law 1: subject_spans → verb (standard subject-verb agreement)
    # Law 2: SOV word order means the argument closest to the verb is
    #   the object (patient); the further one is the agent (subject).
    #   When only one subject_span links to a Law 2 verb, it is the
    #   agent; object detection defers to Step 3a.
    # Source: Slevanayi (2001), pp. 60-61.
    #
    # First pass: Law 1 subject-verb edges
    for subj in subject_spans:
        last_idx = subj["last_idx"]
        for vi, info in verb_info.items():
            if info["law"] != "law1":
                continue
            if 0 < vi - last_idx <= 8 and not _has_clause_boundary_between(tokens, features, last_idx, vi):
                edge_type = "passive_subject_verb" if info.get("passive") else "subject_verb"
                graph.add_edge(
                    subj["indices"][0], vi,
                    edge_type, ["person", "number"],
                    law="law1",
                )

    # Second pass: Law 2 — distinguish agent from object via SOV position
    for vi, info in verb_info.items():
        if info["law"] != "law2":
            continue
        candidates = []
        for subj in subject_spans:
            last_idx = subj["last_idx"]
            if 0 < vi - last_idx <= 8 and not _has_clause_boundary_between(tokens, features, last_idx, vi):
                candidates.append(subj)
        if not candidates:
            continue
        candidates.sort(key=lambda s: s["last_idx"])
        if len(candidates) >= 2:
            # SOV: all but the closest to verb are agents
            for subj in candidates[:-1]:
                graph.add_edge(
                    subj["indices"][0], vi,
                    "agent_non_agreeing", [],
                    law="law2",
                )
                logger.debug(
                    f"Law 2: Agent '{tokens[subj['indices'][0]]}' at "
                    f"{subj['indices'][0]} → verb at {vi} (non-agreeing)"
                )
            # Closest to verb is the object (patient)
            obj = candidates[-1]
            graph.add_edge(
                obj["indices"][0], vi,
                "object_verb_ergative", ["person", "number"],
                law="law2",
            )
            logger.debug(
                f"Law 2: Object '{tokens[obj['indices'][0]]}' at "
                f"{obj['indices'][0]} → verb at {vi} (ergative agreement)"
            )
        else:
            # Single span — agent; object detected in Step 3a
            graph.add_edge(
                candidates[0]["indices"][0], vi,
                "agent_non_agreeing", [],
                law="law2",
            )
            logger.debug(
                f"Law 2: Agent '{tokens[candidates[0]['indices'][0]]}' at "
                f"{candidates[0]['indices'][0]} → verb at {vi} (non-agreeing)"
            )

    # ------------------------------------------------------------------
    # Step 3-VS: Backward subject-verb search (VS / right-extraposition)
    # ------------------------------------------------------------------
    # Sorani canonical order is SOV, but VS (verb-subject) occurs in
    # emphatic, spoken, or presentational constructions. For verbs that
    # have no subject linked yet, search FORWARD for a subject noun.
    # Source: Fattah (1997), pp. 115-116 — VSO variants in Kurdish.
    verbs_with_subject_step3 = set()
    for edge in graph.edges:
        if edge.agreement_type in ("subject_verb", "passive_subject_verb"):
            verbs_with_subject_step3.add(edge.target_idx)
    for vi, info in verb_info.items():
        if vi in verbs_with_subject_step3:
            continue
        if info["law"] != "law1":
            continue
        # Search forward from verb for a subject pronoun or bare noun
        for j in range(vi + 1, min(vi + 8, len(tokens))):
            if _is_clause_boundary(tokens[j], features[j]):
                break
            if tokens[j] in SUBJECT_PRONOUNS:
                p, n = SUBJECT_PRONOUNS[tokens[j]]
                graph.add_edge(
                    j, vi,
                    "backward_subject_verb", ["person", "number"],
                    law="law1",
                )
                logger.debug(
                    "VS order: subject '%s' at %d → verb '%s' at %d",
                    tokens[j], j, tokens[vi], vi,
                )
                break
            if _is_bare_noun(tokens[j], features[j]):
                graph.add_edge(
                    j, vi,
                    "backward_subject_verb", ["person", "number"],
                    law="law1",
                )
                logger.debug(
                    "VS order: bare noun '%s' at %d → verb '%s' at %d",
                    tokens[j], j, tokens[vi], vi,
                )
                break
            # H1 fix: definite/marked nouns in VS order (mirrors Step 2d)
            if (features[j].pos == "NOUN"
                    and not _is_bare_noun(tokens[j], features[j])
                    and not _is_interrogative(tokens[j])
                    and not _is_reciprocal(tokens[j])
                    and getattr(features[j], "case", "") != "obl"
                    and not _is_quantifier(tokens[j])
                    and not _is_mass_noun(tokens[j])
                    and not _is_collective_noun(tokens[j])
                    and not _is_demonstrative(tokens[j])
                    and not _is_proper_noun(tokens[j])):
                graph.add_edge(
                    j, vi,
                    "backward_subject_verb", ["person", "number"],
                    law="law1",
                )
                logger.debug(
                    "VS order: definite noun '%s' at %d → verb '%s' at %d",
                    tokens[j], j, tokens[vi], vi,
                )
                break

    # ------------------------------------------------------------------
    # Step 3a: Detect bare noun OBJECTS in Law 2 that aren't in subject_spans
    # ------------------------------------------------------------------
    # Source: Slevanayi (2001), pp. 55-56; Finding #82, #89
    # In SOV order, a bare noun between the last subject and a Law 2 verb
    # is likely the object (patient). If bare → zero agreement.
    subject_indices = {idx for s in subject_spans for idx in s["indices"]}
    for vi, info in verb_info.items():
        if info["law"] != "law2":
            continue
        # Scan backwards from verb for nouns not already captured.
        # Respect clause boundaries — stop if we hit a conjunction,
        # subordinator, or punctuation that signals a different clause.
        for j in range(vi - 1, max(vi - 10, -1), -1):
            if _is_clause_boundary(tokens[j], features[j]):
                break  # crossed into different clause
            if j in subject_indices:
                continue
            if features[j].pos != "NOUN":
                continue
            # Finding #73: interrogative pronouns are invariant — skip
            if _is_interrogative(tokens[j]):
                continue
            # Finding #74: reciprocal pronouns are invariant objects — skip
            if _is_reciprocal(tokens[j]):
                continue
            # B1: Oblique-marked nouns (after prepositions) do NOT
            # control verb agreement — Slevanayi (2001), pp. 55-56.
            # They receive a traceability-only edge with no features.
            if getattr(features[j], "case", "") == "obl":
                graph.add_edge(
                    j, vi,
                    "oblique_no_agreement", [],
                    law="law2",
                )
                logger.debug(
                    "B1: Oblique noun '%s' at %d → no agreement with "
                    "verb at %d", tokens[j], j, vi,
                )
                break
            if _is_bare_noun(tokens[j], features[j]):
                graph.add_edge(
                    j, vi,
                    "object_verb_ergative_zero", [],
                    law="law2",
                )
                logger.debug(
                    f"Finding #89: Detected bare noun object "
                    f"'{tokens[j]}' at {j} → zero agreement with verb at {vi}"
                )
                break
            else:
                # Non-bare noun object → standard ergative agreement
                graph.add_edge(
                    j, vi,
                    "object_verb_ergative", ["person", "number"],
                    law="law2",
                )
                break
    # ------------------------------------------------------------------
    # Step 3b: Pro-drop recovery (Finding #75 — subject omission)
    # ------------------------------------------------------------------
    # Sorani Kurdish allows pro-drop: the subject pronoun is omitted
    # when verb inflection unambiguously marks person/number.
    # Source: Slevanayi (2001), pp. 61-62 — verb agreement suffixes
    # encode the dropped subject.
    #
    # For each verb with NO subject linked, create a self-referencing
    # "pro_drop_agreement" edge that records the verb's own inflected
    # person/number as the implicit subject agreement.
    verbs_with_subject = set()
    for edge in graph.edges:
        if edge.agreement_type in ("subject_verb", "agent_non_agreeing"):
            verbs_with_subject.add(edge.target_idx)
    for vi, info in verb_info.items():
        if vi in verbs_with_subject:
            continue
        # Use the verb's own inflection as the agreement source.
        v_person = features[vi].person
        v_number = features[vi].number
        if v_person:
            graph.add_edge(
                vi, vi,
                "pro_drop_agreement", ["person", "number"],
                law=info["law"],
            )
            logger.debug(
                "Pro-drop at %d: verb '%s' inflects %s%s — no overt subject",
                vi, tokens[vi], v_person, v_number or "",
            )

    # ------------------------------------------------------------------
    # Step 4: NP-internal agreement (ezafe constructions)
    # ------------------------------------------------------------------
    # Slevanayi (2001), pp. 37-48: determiners agree with head nouns, but
    # adjectives are invariant (never agree in number/gender).
    # Quantifiers always trigger plural on the verb (pp. 87-88), but the
    # noun after a numeral stays SINGULAR (Maaruf 2010, p. 139):
    #   "دوو کوڕ" not *"دوو کوڕان"
    # Mukriani (2000, pp. 24-26): only PRE-NOMINAL quantifiers trigger
    # plural verb agreement.  Post-nominal ordinals (e.g. کەسی دووەم)
    # do NOT control verb agreement — the noun controls instead.
    #
    # Mass noun + measure word routing (Slevanayi 2001, pp. 46-47, 57):
    # When a mass noun appears with a measure word, the agreement graph
    # routes through the measure word — the mass noun never controls
    # verb agreement directly.
    #
    # Collective noun handling (Slevanayi 2001, pp. 45-46, 58):
    # Bare collective → singular verb; with هەموو → plural verb.
    #
    # Demonstrative dual behavior (Slevanayi 2001, pp. 83-86):
    # Pro-form (standalone ئەوە/ئەمە) → direct verb agreement edge;
    # Determiner (ئەو/ئەم + noun) → NP-internal edge only.
    for i in range(len(tokens) - 1):
        if _is_quantifier(tokens[i]):
            # Quantifier → noun: no number agreement edge (noun stays singular)
            # But  quantifier → verb: forces plural agreement
            # Only add edge if quantifier is in pre-nominal position
            # (Mukriani 2000, pp. 24-26)
            next_tok = tokens[i + 1] if i + 1 < len(tokens) else ""
            # Positive POS check: next token should be nominal (NOUN/ADJ)
            # or have no POS yet (unrecognized open-class word, likely noun).
            # Avoids false edges when quantifier precedes ADP, ADV, etc.
            next_pos = features[i + 1].pos if i + 1 < len(features) else ""
            is_prenominal = (
                next_tok not in SUBJECT_PRONOUNS
                and not any(next_tok.startswith(p) for p in ALL_VERB_PREFIXES)
                and next_tok != "و"
                and next_pos in ("", "NOUN", "ADJ", "DET")
            )
            if is_prenominal:
                for vi, info in verb_info.items():
                    if 0 < vi - i <= 10 and not _has_clause_boundary_between(tokens, features, i, vi):
                        graph.add_edge(
                            i, vi,
                            "quantifier_verb", ["number"],
                            law="law1",
                        )
                        break
        elif _is_measure_word(tokens[i]):
            # Measure word controls verb agreement, not the mass noun
            # (Slevanayi 2001, pp. 46-47, 57)
            for vi, info in verb_info.items():
                if 0 < vi - i <= 10 and not _has_clause_boundary_between(tokens, features, i, vi):
                    graph.add_edge(
                        i, vi,
                        "measure_word_verb", ["number"],
                        law="law1",
                    )
                    break
        elif _is_mass_noun(tokens[i]):
            # F#68: Mass noun without measure word → no verb agreement.
            # Mass nouns require a measure word for quantification;
            # the measure word controls agreement, not the mass noun.
            # Source: Slevanayi (2001), pp. 46-47, 53, 57
            has_measure = (
                i + 1 < len(tokens) and _is_measure_word(tokens[i + 1])
            )
            if not has_measure:
                for vi, info in verb_info.items():
                    if 0 < vi - i <= 10 and not _has_clause_boundary_between(tokens, features, i, vi):
                        graph.add_edge(
                            i, vi,
                            "mass_noun_no_agreement", [],
                            law="law1",
                        )
                        logger.debug(
                            f"F#68: Mass noun '{tokens[i]}' at {i} — "
                            f"no measure word, no agreement with verb at {vi}"
                        )
                        break
        elif _is_collective_noun(tokens[i]):
            # Collective noun dual behavior (Slevanayi 2001, pp. 45-46, 58)
            # Finding #69: morphologically singular, semantically plural.
            # Bare collective → singular verb (3sg default)
            # With هەموو/گشت preceding → plural verb
            has_universal_quantifier = (
                i > 0 and tokens[i - 1] in {"هەموو", "گشت"}
            )
            for vi, info in verb_info.items():
                if 0 < vi - i <= 10 and not _has_clause_boundary_between(tokens, features, i, vi):
                    if has_universal_quantifier:
                        # هەموو + collective → plural verb
                        graph.add_edge(
                            i, vi,
                            "collective_plural", ["number"],
                            law="law1",
                        )
                        logger.debug(
                            f"Collective+quantifier → plural edge: "
                            f"{tokens[i]} at {i} → verb at {vi}"
                        )
                    else:
                        # Bare collective → singular (3sg default, person only)
                        graph.add_edge(
                            i, vi,
                            "collective_singular", ["person"],
                            law="law1",
                        )
                        logger.debug(
                            f"Bare collective → singular edge: "
                            f"{tokens[i]} at {i} → verb at {vi}"
                        )
                    break
        elif _is_demonstrative(tokens[i]):
            next_tok = tokens[i + 1] if i + 1 < len(tokens) else ""
            # Demonstrative as determiner: next token is a noun/adjective
            # (not a verb, preposition, conjunction, pronoun, or boundary)
            next_is_nominal = (
                next_tok
                and not _is_invariant(next_tok)
                and next_tok not in SUBJECT_PRONOUNS
                and features[i + 1].pos not in ("VERB", "ADP", "CONJ", "PRON", "PUNCT")
                and not any(next_tok.startswith(p) for p in ALL_VERB_PREFIXES)
            ) if i + 1 < len(features) else False
            if next_is_nominal:
                # Demonstrative as determiner (in NP): NP-internal edge only
                # (Slevanayi 2001, pp. 83-86)
                graph.add_edge(i, i + 1, "dem_det_noun", ["number"])
            else:
                # Demonstrative as pro-form (standalone): verb agreement edge
                # (Slevanayi 2001, pp. 83-86)
                for vi, info in verb_info.items():
                    if 0 < vi - i <= 10 and not _has_clause_boundary_between(tokens, features, i, vi):
                        graph.add_edge(
                            i, vi,
                            "dem_proform_verb", ["person", "number"],
                            law=info["law"],
                        )
                        break
        elif tokens[i].endswith("ی") and len(tokens[i]) > 1:
            modifier = tokens[i + 1]
            if _is_invariant(modifier):
                # Adjective/reflexive/etc. — no agreement edge (invariant)
                logger.debug(
                    f"Skipping invariant modifier '{modifier}' at position {i + 1}"
                )
            elif features[i + 1].pos == "ADJ":
                # Finding #79: adjectives NEVER agree in number/gender.
                # Mark with explicit invariant edge for traceability;
                # the model can learn to ignore this link.
                graph.add_edge(
                    i, i + 1,
                    "adjective_invariant", [],
                )
                logger.debug(
                    f"F#79: Adjective invariant edge for "
                    f"'{modifier}' at position {i + 1}"
                )
            elif features[i + 1].pos in ("", "NOUN", "DET", "NUM"):
                graph.add_edge(i, i + 1, "noun_det", ["number"])

    # ------------------------------------------------------------------
    # Step 5: Clitic agreement edges — with Set 1/2/3 distinction
    # ------------------------------------------------------------------
    # Source: Slevanayi (2001), pp. 34-37; Finding #9, #22
    # Haji Marif (2014) — Set 1 vs Set 2 vs Set 3 role distinction.
    #
    # Set 1 clitics (م/ت/ی/مان/تان/یان): bound pronouns that
    #   reference an argument. Role flips by tense (ergativity):
    #   - Present tense / intransitive past: agent
    #   - Past transitive: patient
    #
    # Set 2 clitics (م/یت/ێت/ین/ن): agreement suffixes on verbs,
    #   marking person/number of the agreement controller. These are
    #   already captured by verb person suffix extraction in the
    #   analyzer — do NOT create separate clitic edges for Set 2.
    #
    # Set 3 (possessive): on non-verb hosts, NEVER triggers agreement.
    for i, tok in enumerate(tokens):
        # Skip tokens with ezafe case — trailing ی is ezafe, not
        # a possessive clitic (e.g. کوڕی گەورە = boy's big).
        if features[i].case == "ez":
            continue
        for cl in CLITIC_FORMS:
            if tok.endswith(cl) and len(tok) > len(cl) + 1:
                # CLITIC_FORMS contains only Set 1 clitics (bound pronouns).
                # Set 2 clitics (verb agreement suffixes like یت/ێت/ین/ن)
                # are handled by the analyzer's person/number extraction
                # and are not iterated here.

                # F#71: Possessive clitics on non-verb hosts NEVER agree.
                # If the host token is not a verb, this is a possessive
                # clitic (Set 3) and should not create agreement edges.
                host_is_verb = (
                    features[i].pos == "VERB"
                    or any(tok.startswith(p) for p in ALL_VERB_PREFIXES)
                    or i in verb_info
                )
                if not host_is_verb:
                    # F#71: Possessive clitic on non-verb host — informational
                    # edge so the model can distinguish possessives from
                    # agreement clitics. No feature matching required.
                    graph.add_edge(
                        i, i, "possessive_no_agreement", [],
                    )
                    logger.debug(
                        f"F#71: Possessive clitic '{cl}' on non-verb "
                        f"'{tok}' at {i} — possessive_no_agreement edge"
                    )
                    break

                # Find the nearest verb to decide tense/transitivity
                nearest_verb_idx = None
                nearest_dist = 999
                for vi in verb_info:
                    dist = abs(vi - i)
                    if dist < nearest_dist:
                        nearest_dist = dist
                        nearest_verb_idx = vi

                # Determine clitic role from verb context
                if nearest_verb_idx is not None:
                    vinfo = verb_info[nearest_verb_idx]
                    # F#101: Wistin exception — always uses law1 (agent)
                    # regardless of tense, because ویستن agrees with
                    # the subject even though it's transitive.
                    if vinfo.get("wistin_exception"):
                        clitic_role = "clitic_agent"
                    elif vinfo["law"] == "law2":
                        # Past transitive (ergative): Set 1 clitic = patient
                        clitic_role = "clitic_patient"
                    else:
                        # Present / intransitive past: Set 1 clitic = agent
                        clitic_role = "clitic_agent"
                else:
                    clitic_role = "clitic_agent"  # default

                # Link to nearest preceding pronoun or noun (within 6 tokens)
                for j in range(i - 1, max(i - 6, -1), -1):
                    if _is_clause_boundary(tokens[j], features[j]):
                        break  # don't cross clause boundaries
                    if (tokens[j] in SUBJECT_PRONOUNS
                            or features[j].pos in ("NOUN", "PRON")
                            or tokens[j] in DEMONSTRATIVES
                            or tokens[j] in COMMON_PROPER_NOUNS):
                        graph.add_edge(
                            j, i, clitic_role, ["person", "number"],
                            law=verb_info[nearest_verb_idx]["law"] if nearest_verb_idx is not None else "",
                        )
                        logger.debug(
                            f"Clitic edge: {clitic_role} from {j}→{i}, "
                            f"clitic='{cl}' on '{tok}'"
                        )
                        break
                break

    # ------------------------------------------------------------------
    # Step 6: Existential هەبوون three-way distinction
    # ------------------------------------------------------------------
    # Source: Slevanayi (2001), pp. 75-77
    # When بوون/هەبوون appears, classify its use:
    #   Existence (one argument): agreement set agrees normally
    #   Possession (two arguments): possessive set NEVER agrees — 3sg
    for vi, tok in enumerate(tokens):
        if _is_existential_verb(tok):
            # Check if there's a possessive clitic preceding (possession use)
            has_possessive = False
            for j in range(vi - 1, max(vi - 8, -1), -1):
                for poss in INVARIANT_POSSESSIVES:
                    if tokens[j].endswith(poss) and len(tokens[j]) > len(poss):
                        has_possessive = True
                        break
                if has_possessive:
                    break

            if has_possessive:
                # Possession: verb stays 3sg — no agreement edge to possessor
                logger.debug(
                    f"Existential-possession at {vi}: verb stays 3sg, "
                    f"possessive pronoun does not agree"
                )
            elif vi not in verb_info:
                # Existence use: add to verb_info as Law 1 intransitive
                verb_info[vi] = {
                    "tense": "past" if "بوو" in tok else "present",
                    "transitive": False,
                    "law": "law1",
                }

    # ------------------------------------------------------------------
    # Step 7: Vocative-imperative number agreement
    # ------------------------------------------------------------------
    # Source: Slevanayi (2001), pp. 16, 72-73
    # Vocative noun phrase agrees with imperative verb in NUMBER:
    #   کوڕۆ وەرە (sg.VOC + sg.IMP) vs. کوڕینۆ وەرن (pl.VOC + pl.IMP)
    for i, tok in enumerate(tokens):
        # Skip tokens already identified as verbs — they can't be
        # vocative nouns even if their surface form ends in ۆ/ێ.
        if i in verb_info:
            continue
        if _is_vocative(tok, features[i]):
            # Find the nearest imperative verb within the same clause
            for vi, info in verb_info.items():
                if (info.get("tense") == "imperative"
                        and 0 < abs(vi - i) <= 6
                        and not _has_clause_boundary_between(tokens, features, i, vi)
                        and any(tokens[vi].startswith(p) for p in {"ب", "مە"})):
                    graph.add_edge(
                        i, vi,
                        "vocative_imperative", ["number"],
                        law="law1",
                    )
                    break

    # ------------------------------------------------------------------
    # Step 8: Relative clause agreement edges
    # ------------------------------------------------------------------
    # Source: Slevanayi (2001); Finding #141
    # Pattern: definite-noun + ezafe + کە + ... + verb
    # The relative clause verb should agree with the antecedent noun.
    # Detecting: find کە tokens, look left for the antecedent (definite
    # noun with ezafe), look right for the nearest verb in the clause.
    # Clause scope: stop at the next کە, و+verb, or punctuation to
    # avoid misattaching in multi-clause sentences.
    for i, tok in enumerate(tokens):
        if tok != RELATIVE_CLAUSE_MARKER:
            continue
        # Find antecedent: the nearest preceding noun (often has ezafe).
        # Window of 4 positions handles modified NPs like
        # "noun + adj + adj + کە" where the head noun is up to 4 tokens back.
        antecedent_idx = None
        for j in range(i - 1, max(i - 5, -1), -1):
            if features[j].pos in ("NOUN", "PRON"):
                antecedent_idx = j
                break
        if antecedent_idx is None:
            continue
        # Find relative clause verb: nearest following verb within the
        # same clause. Stop at clause boundaries.
        rel_verb_idx = None
        for j in range(i + 1, min(i + 10, len(tokens))):
            if tokens[j] == RELATIVE_CLAUSE_MARKER:
                break  # nested relative clause — stop
            if _is_clause_boundary(tokens[j], features[j]):
                break  # any clause boundary — stop
            if j in verb_info or features[j].pos == "VERB":
                rel_verb_idx = j
                break
        if rel_verb_idx is None:
            continue
        graph.add_edge(
            antecedent_idx, rel_verb_idx,
            "relative_clause", ["person", "number"],
            law=verb_info.get(rel_verb_idx, {}).get("law", "law1"),
        )
        logger.debug(
            f"F#141: Relative clause edge: antecedent '{tokens[antecedent_idx]}' "
            f"at {antecedent_idx} → verb '{tokens[rel_verb_idx]}' at {rel_verb_idx}"
        )

    # ------------------------------------------------------------------
    # Step 4a: Pre-head determiner validation (F#117)
    # ------------------------------------------------------------------
    # Source: Farhadi (2013), pp. 15-16
    # Pre-head determiners attach directly before the head noun WITHOUT
    # ezafe. Inserting ezafe between pre-head and head is an error
    # (*زۆری کتێب → زۆر کتێب).
    # Flag violations for the error correction model.
    for i, tok in enumerate(tokens):
        # Match both bare determiners ("زۆر") and those with spurious ezafe ("زۆری")
        is_prehead = tok in PRE_HEAD_DETERMINERS
        has_spurious_ezafe = (
            tok.endswith("ی") and len(tok) > 1 and tok[:-1] in PRE_HEAD_DETERMINERS
        )
        if (is_prehead or has_spurious_ezafe) and i + 1 < len(tokens):
            next_tok = tokens[i + 1]
            if has_spurious_ezafe and features[i + 1].pos in ("", "NOUN", "DET", "NUM"):
                # e.g. "زۆری" before noun — spurious ezafe on pre-head
                graph.add_edge(
                    i, i + 1, "noun_det", ["number"],
                )
                logger.debug(
                    f"F#117: Pre-head '{tok}' at {i} has potential "
                    f"spurious ezafe before '{next_tok}'"
                )

    # ------------------------------------------------------------------
    # Step 4b: Definite marker migration validation (F#115)
    # ------------------------------------------------------------------
    # Source: Farhadi (2013), pp. 16-17
    # In descriptive NPs (noun + adj modifier), ەکە migrates to the
    # LAST modifier, not on the head noun.
    # In possessive NPs, ەکە stays on the head noun.
    if DEFINITE_MARKER_MIGRATION_DESCRIPTIVE:
        for i, tok in enumerate(tokens):
            if features[i].pos == "NOUN" and features[i].definiteness == "def":
                # Check if followed by ezafe + adjective (descriptive NP)
                if (i + 1 < len(tokens)
                        and features[i + 1].pos == "ADJ"):
                    # ەکە should be on the adjective, not the noun
                    logger.debug(
                        f"F#115: Definite noun '{tok}' at {i} followed by "
                        f"ADJ '{tokens[i + 1]}' — marker should migrate to modifier"
                    )

    # ------------------------------------------------------------------
    # Step 4c: ی/یی ezafe allomorph validation (F#165)
    # ------------------------------------------------------------------
    # Source: Rasul (2004), pp. 126-127
    # Consonant-final nouns take single ی ezafe; vowel-final (especially
    # ی-final) nouns require یی.  This step flags tokens where a single
    # ی appears after a ی-final base but should be doubled.
    _yi = "ی"
    for i, tok in enumerate(tokens):
        if len(tok) < 3:
            continue
        # Only check potential ezafe contexts (followed by modifier)
        if i + 1 >= len(tokens):
            continue
        next_pos = features[i + 1].pos
        if next_pos not in ("ADJ", "NOUN", "PRON"):
            continue
        base = tok[:-1] if tok.endswith(_yi) else None
        if base is None:
            continue
        # If base itself ends with ی, the surface form should be یی
        # This corresponds to YI_DOUBLE_SCENARIOS[1] ("yi_final_plus_ezafe")
        if base.endswith(_yi) and not tok.endswith("یی"):
            logger.debug(
                f"F#165: '{tok}' at {i} — ی-final base '{base}' "
                f"needs double یی (scenario: {YI_DOUBLE_SCENARIOS[1]})"
            )

    # ------------------------------------------------------------------
    # Step 9: Complement-requiring verb validation (F#118)
    # ------------------------------------------------------------------
    # Source: Farhadi (2013), pp. 31-33
    # Verbs in COMPLEMENT_REQUIRING_VERBS need a prepositional/locative
    # complement within the clause. Flag when complement is absent.
    for vi, info in verb_info.items():
        verb_token = tokens[vi]
        # Check if any stem of complement-requiring verbs matches
        for comp_stem in COMPLEMENT_REQUIRING_VERBS:
            if comp_stem in verb_token:
                # Scan nearby tokens for a preposition (complement evidence)
                has_complement = False
                for j in range(max(0, vi - 10), min(len(tokens), vi + 5)):
                    if features[j].pos == "ADP":
                        has_complement = True
                        break
                if not has_complement:
                    logger.debug(
                        f"F#118: Complement-requiring verb '{verb_token}' "
                        f"at {vi} has no prepositional complement detected"
                    )
                break

    # ------------------------------------------------------------------
    # Step 10: هەرگیز/هیچ tense restriction (F#256)
    # ------------------------------------------------------------------
    # Negation adverbs هەرگیز, هیچ ڕەنگێ, هیچ کلۆجێک must pair with
    # past or future tense verbs. Present-tense copular/stative context
    # is ungrammatical (KSA 2011, p. 251).
    for i, tok in enumerate(tokens):
        if tok not in HERGIZ_ADVERBS:
            continue
        # Find the nearest verb in the clause
        for vi, info in verb_info.items():
            if abs(vi - i) > 6:
                continue
            if _has_clause_boundary_between(tokens, features, i, vi):
                continue
            if info["tense"] in HERGIZ_BANNED_TENSES:
                graph.add_edge(
                    source=i, target=vi,
                    agreement_type="adverb_verb_tense",
                    features=["tense"],
                )
                logger.debug(
                    "F#256: HERGIZ adverb '%s' at %d with present-tense "
                    "verb '%s' at %d — tense restriction violated",
                    tok, i, tokens[vi], vi,
                )
            break

    # ------------------------------------------------------------------
    # Step 11: Conditional marker → verb mood agreement
    # ------------------------------------------------------------------
    # Source: Maaruf (2009), pp. 121-122 (Finding #255)
    # مەگەر requires subjunctive mood only.
    # ئەگەر / گەر accept both subjunctive and indicative.
    # The conditional marker must link to the nearest verb in its clause
    # so the error‐correction model can flag mood mismatches.
    for i, tok in enumerate(tokens):
        if tok not in CLAUSE_INITIAL_ONLY_CONJUNCTIONS:
            continue
        # Find the nearest verb after the conditional marker
        for vi, info in verb_info.items():
            if vi <= i:
                continue
            if vi - i > 8:
                break
            if _has_clause_boundary_between(tokens, features, i, vi):
                continue
            graph.add_edge(
                source=i, target=vi,
                agreement_type="conditional_agreement",
                features=["tense"],
            )
            logger.debug(
                "B4: Conditional marker '%s' at %d → verb '%s' at %d",
                tok, i, tokens[vi], vi,
            )
            break

    logger.debug(
        f"Built agreement graph with {len(graph.edges)} edges for "
        f"{len(tokens)} tokens"
    )
    return graph
