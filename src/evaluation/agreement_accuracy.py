"""
Sorani Kurdish Agreement Accuracy Checker

Sorani-specific evaluation metric that checks whether agreement constraints
are satisfied in the model's output. Measures the percentage of sentences
where targeted agreement checks pass.

Uses the rule-based morphological analyzer (Amin 2016, Fatah & Qadir 2006)
and agreement constants from Slevanayi (2001) to detect violations.
"""

import logging
from dataclasses import dataclass
from typing import Optional

from ..morphology.analyzer import (
    MorphologicalAnalyzer,
    CLITIC_PERSON_MAP,
)
from ..morphology.constants import SUBJECT_PRONOUNS, TRANSITIVE_PAST_STEMS
from ..morphology.builder import (
    _is_present_verb,
    _is_transitive_past,
)

logger = logging.getLogger(__name__)

# Subject pronouns → expected verb person/number
# Canonical source: SUBJECT_PRONOUNS from constants.py (Amin 2016, pp. 17-18)
_PRONOUN_AGREEMENT = SUBJECT_PRONOUNS

# Present-tense verb ending → (person, number)
# These are Set 2 agreement suffixes on verbs (NOT Set 1 clitics).
# Used by _check_subject_verb() to extract verb person/number.
# Source: Amin (2016), pp. 17-18
_PRESENT_ENDINGS: dict[str, tuple[str, str]] = {
    "م":   ("1", "sg"),   # Set 2: 1sg
    "ەم":  ("1", "sg"),   # Set 2: 1sg (epenthetic)
    "یت":  ("2", "sg"),   # Set 2: 2sg
    "ێت":  ("3", "sg"),   # Set 2: 3sg
    "ات":  ("3", "sg"),   # Set 2: 3sg (after -a stems)
    "ێ":   ("3", "sg"),   # Set 2: 3sg (short form)
    "ین":  ("1", "pl"),   # Set 2: 1pl
    "ن":   ("3", "pl"),   # Set 2: 3pl
    "ەن":  ("3", "pl"),   # Set 2: 3pl (epenthetic)
}

# Verb prefixes that indicate tense
_PRESENT_PREFIXES = ("دە", "ئە")
_NEGATION_PRESENT_PREFIX = "نا"
_NEGATION_PAST_PREFIX = "نە"
_IMPERATIVE_PREFIX = "ب"


@dataclass
class AgreementResult:
    """Result of agreement checking on a single sentence."""
    sentence: str
    checks_passed: int
    checks_total: int
    violations: list[str]
    
    @property
    def accuracy(self) -> float:
        return self.checks_passed / self.checks_total if self.checks_total > 0 else 1.0
    
    @property
    def is_correct(self) -> bool:
        return len(self.violations) == 0


class AgreementChecker:
    """Check Sorani Kurdish agreement constraints in sentences.
    
    Uses rule-based morphological analysis to detect:
    1. Subject-verb person/number mismatch (Law 1 — Slevanayi 2001, p. 89)
    2. Clitic person consistency (F#9, F#133 — same-set exclusion)
    3. Ezafe presence/absence (F#165 — ی/یی six scenarios)
    4. Tense marker consistency within a clause (F#254 — coordination tense)
    
    Background agreement findings informing these checks:
      F#81  — Case determines agreement domain (nom ↔ external, obl ↔ internal)
      F#177 — Sorani lacks morphological case marking (cross-dialectal)
      F#205 — Inanimate nouns carry arbitrary grammatical gender
      F#206 — Post-head determiners always in nominative case internally
    """
    
    def __init__(self, analyzer: Optional[MorphologicalAnalyzer] = None):
        self._analyzer = analyzer or MorphologicalAnalyzer(use_klpt=False)
    
    def check_sentence(self, sentence: str) -> AgreementResult:
        """Run all agreement checks on a sentence."""
        violations = []
        failed_checks = 0
        total_checks = 0
        
        # Check 1: Subject-verb number
        sv_violations = self._check_subject_verb(sentence)
        violations.extend(sv_violations)
        if sv_violations:
            failed_checks += 1
        total_checks += 1
        
        # Check 2: Clitic consistency
        cl_violations = self._check_clitic_consistency(sentence)
        violations.extend(cl_violations)
        if cl_violations:
            failed_checks += 1
        total_checks += 1
        
        # Check 3: Ezafe presence
        ez_violations = self._check_ezafe(sentence)
        violations.extend(ez_violations)
        if ez_violations:
            failed_checks += 1
        total_checks += 1
        
        # Check 4: Tense consistency
        t_violations = self._check_tense_consistency(sentence)
        violations.extend(t_violations)
        if t_violations:
            failed_checks += 1
        total_checks += 1

        # Check 5: Object-verb agreement (Law 2 — ergative past transitive)
        ov_violations = self._check_object_verb_ergative(sentence)
        violations.extend(ov_violations)
        if ov_violations:
            failed_checks += 1
        total_checks += 1

        # Check 6: Negative concord (double negation required in Sorani)
        nc_violations = self._check_negative_concord(sentence)
        violations.extend(nc_violations)
        if nc_violations:
            failed_checks += 1
        total_checks += 1

        # Check 7: Orthographic consistency (common confusion pairs)
        orth_violations = self._check_orthography(sentence)
        violations.extend(orth_violations)
        if orth_violations:
            failed_checks += 1
        total_checks += 1

        # Check 8: Conditional agreement (ئەگەر clause tense constraints)
        cond_violations = self._check_conditional_agreement(sentence)
        violations.extend(cond_violations)
        if cond_violations:
            failed_checks += 1
        total_checks += 1

        # Check 9: Quantifier–noun agreement (CRIT-4)
        qn_violations = self._check_quantifier_noun(sentence)
        violations.extend(qn_violations)
        if qn_violations:
            failed_checks += 1
        total_checks += 1

        # Check 10: Relative clause agreement (CRIT-4)
        rc_violations = self._check_relative_clause(sentence)
        violations.extend(rc_violations)
        if rc_violations:
            failed_checks += 1
        total_checks += 1

        # Check 11: Vocative–imperative agreement (CRIT-4)
        vi_violations = self._check_vocative_imperative(sentence)
        violations.extend(vi_violations)
        if vi_violations:
            failed_checks += 1
        total_checks += 1

        # Check 12: Adverb–verb tense consistency (CRIT-4)
        av_violations = self._check_adverb_verb_tense(sentence)
        violations.extend(av_violations)
        if av_violations:
            failed_checks += 1
        total_checks += 1

        # Check 13: Compound subject person resolution (Slevanayi 2001, p. 89)
        cs_violations = self._check_compound_subject(sentence)
        violations.extend(cs_violations)
        if cs_violations:
            failed_checks += 1
        total_checks += 1

        # Check 14: Bare noun person-only agreement (Slevanayi 2001, p. 60)
        bn_violations = self._check_bare_noun_agreement(sentence)
        violations.extend(bn_violations)
        if bn_violations:
            failed_checks += 1
        total_checks += 1
        
        passed = total_checks - failed_checks
        
        return AgreementResult(
            sentence=sentence,
            checks_passed=passed,
            checks_total=total_checks,
            violations=violations,
        )
    
    def _clause_boundary_indices(self, words: list[str]) -> list[int]:
        """Return word indices that are clause boundaries.

        A clause boundary occurs at و when the preceding clause segment
        contains verb evidence OR when the segment is long enough to be
        a verbless clause (nominal/prepositional predicate). Punctuation
        (،/./?/!) also marks boundaries.
        """
        verb_prefixes = ("دە", "ئە", "نا", "بی", "بە", "ب")
        boundaries: list[int] = []
        segment_has_verb = False
        segment_word_count = 0
        for i, word in enumerate(words):
            if word in ("،", ".", "؟", "!"):
                boundaries.append(i)
                segment_has_verb = False
                segment_word_count = 0
            elif word == "و" and i > 0:
                if segment_has_verb or segment_word_count >= 2:
                    boundaries.append(i)
                    segment_has_verb = False
                    segment_word_count = 0
                # else: single-word NP-internal و (e.g. "نان و پەنیر")
            else:
                segment_word_count += 1
                if any(word.startswith(vp) for vp in verb_prefixes):
                    segment_has_verb = True
                for stem in TRANSITIVE_PAST_STEMS:
                    if word.startswith(stem):
                        segment_has_verb = True
                        break
        return boundaries

    def _check_subject_verb(self, sentence: str) -> list[str]:
        """Check subject-verb person/number agreement (Law 1).
        
        Source: Slevanayi (2001), p. 89 — verb agrees with subject in
        person and number for intransitive and present-tense transitive.

        Scans forward from each pronoun to the next clause boundary
        (instead of a fixed 6-word window) to avoid missing distant verbs.
        """
        violations = []
        words = self._analyzer.tokenize(sentence)
        clause_bounds = set(self._clause_boundary_indices(words))
        
        for i, word in enumerate(words):
            if word not in _PRONOUN_AGREEMENT:
                continue
            expected_person, expected_number = _PRONOUN_AGREEMENT[word]
            
            # Scan forward for a verb until clause boundary
            for j in range(i + 1, len(words)):
                if j in clause_bounds:
                    break
                candidate = words[j]
                is_present = any(candidate.startswith(p) for p in _PRESENT_PREFIXES)
                is_neg_present = candidate.startswith(_NEGATION_PRESENT_PREFIX)
                
                if not (is_present or is_neg_present):
                    continue
                
                # Extract verb ending → person/number
                verb_pn = self._verb_ending_to_pn(candidate)
                if verb_pn is None:
                    break
                
                verb_person, verb_number = verb_pn
                if verb_person != expected_person or verb_number != expected_number:
                    violations.append(
                        f"Subject-verb mismatch: '{word}' ({expected_person}{expected_number}) "
                        f"with verb '{candidate}' ({verb_person}{verb_number})"
                    )
                break
        
        return violations
    
    @staticmethod
    def _verb_ending_to_pn(verb: str) -> Optional[tuple[str, str]]:
        """Extract person/number from a verb's ending suffix."""
        # Check longest suffixes first
        for suffix in sorted(_PRESENT_ENDINGS, key=len, reverse=True):
            if verb.endswith(suffix):
                return _PRESENT_ENDINGS[suffix]
        return None
    
    def _check_clitic_consistency(self, sentence: str) -> list[str]:
        """Check for inconsistent clitic usage within a clause.

        Rules enforced:
        - F#133: Same-set clitic exclusion — two clitics from the same set
          (e.g., two Set 1 clitics) cannot co-occur in a simple sentence.
        - F#9: Clitic person must be plausible in context (no two different
          person clitics on adjacent words unless compounding).

        EVAL-5 fixes:
        - Past verbs also carry Set 2 suffixes (not Set 1 clitics); skip them.
        - Uses analyzer's morphological features for ی disambiguation instead
          of bare endswith("ی"), avoiding false filtering on ezafe/indefinite ی.
        """
        violations = []
        words = self._analyzer.tokenize(sentence)
        found_clitics: list[tuple[str, str, str]] = []  # (clitic, person, number)

        for word in words:
            # Set 2 verb suffixes (م/ت/ی on دەکەم etc.) are NOT
            # Set 1 clitics. Skip present-tense verbs entirely.
            if _is_present_verb(word) or word.startswith(_NEGATION_PRESENT_PREFIX):
                continue
            # Also skip past verbs — their suffixes are Set 2 agreement,
            # not Set 1 clitics (EVAL-5 fix: prevents Set 2 leak).
            if _is_transitive_past(word):
                continue
            for cl, (person, number) in CLITIC_PERSON_MAP.items():
                if word.endswith(cl) and len(word) > len(cl) + 1:
                    stem = word[: -len(cl)]
                    # Use analyzer's morphological features to detect
                    # possessive (Set 3) constructions. The old bare
                    # endswith("ی") check was ambiguous (EVAL-5 fix).
                    feats = self._analyzer.analyze_token(stem + "ی") if cl != "ی" else self._analyzer.analyze_token(word)
                    if cl != "ی" and feats.case == "ez":
                        # Stem has ezafe case → this is possessive, not Set 1
                        continue
                    if cl == "ی" and feats.raw_analysis.get("yi_ambiguous"):
                        # Analyzer flagged ی as ambiguous (likely ezafe/possessive)
                        continue
                    found_clitics.append((cl, person, number))
                    break

        # Same-set exclusion check (F#133): two Set 1 clitics cannot
        # co-occur in a simple sentence. Flag if we see two or more
        # distinct clitics — either with different persons or the same
        # clitic appearing on multiple hosts.
        if len(found_clitics) >= 2:
            persons_seen = {c[1] for c in found_clitics}
            distinct_clitics = {c[0] for c in found_clitics}
            if len(persons_seen) >= 2 or len(distinct_clitics) >= 2 or len(found_clitics) > len(distinct_clitics):
                violations.append(
                    f"Clitic inconsistency: {len(found_clitics)} Set 1 clitics "
                    f"with {len(distinct_clitics)} distinct forms and "
                    f"{len(persons_seen)} person(s) in one clause (F#133)"
                )

        return violations
    
    def _check_ezafe(self, sentence: str) -> list[str]:
        """Check for ezafe (ی/یی) issues in noun phrases.
        
        Rules enforced:
        - F#165: After consonant-final noun, ezafe is ی; after vowel-final, یی.
        - F#10/R4: Demonstrative (ئەم/ئەو) cannot co-occur with ەکە/ێک.
        - Missing ezafe between noun and attributive adjective is an error.
        """
        violations = []
        words = self._analyzer.tokenize(sentence)
        
        # Check demonstrative + definiteness co-occurrence (F#10, Rule R4)
        demonstratives = {"ئەم", "ئەو", "ئەمە", "ئەوە"}
        definite_markers = ("ەکە", "یەکە", "ەکان", "یەکان")
        indefinite_markers = ("ێک", "یەک", "ێکی")
        
        in_dem_np = False
        dem_word = ""
        dem_words_seen = 0
        for i, word in enumerate(words):
            if word in demonstratives:
                in_dem_np = True
                dem_word = word
                dem_words_seen = 0
                continue
            
            if in_dem_np:
                dem_words_seen += 1
                # Within a demonstrative NP, check for prohibited markers
                has_def = any(word.endswith(m) for m in definite_markers)
                has_indef = any(word.endswith(m) for m in indefinite_markers)
                if has_def:
                    violations.append(
                        f"Demonstrative+definite co-occurrence: '{dem_word}' with "
                        f"definite marker on '{word}' (F#10/R4)"
                    )
                if has_indef:
                    violations.append(
                        f"Demonstrative+indefinite co-occurrence: '{dem_word}' with "
                        f"indefinite marker on '{word}' (F#10/R4)"
                    )
                # Demonstrative NPs in Sorani can span up to ~3 words
                # (dem + adj* + noun + closing -ə). End tracking after
                # 3 non-demonstrative words or at a clause boundary.
                if dem_words_seen >= 3 or word in {"و", "،", ".", "؟"}:
                    in_dem_np = False
        
        # Check ezafe allomorph: ی after consonant, یی after vowel (F#165)
        # Kurdish vowel characters at word-final position
        _vowels = {"ا", "ە", "ێ", "ی", "ۆ", "و"}
        for i, word in enumerate(words):
            if i + 1 >= len(words):
                continue
            # Check if word ends with ezafe ی and is followed by a modifier
            if not word.endswith("ی") or len(word) < 2:
                continue
            base = word[:-1]
            if not base:
                continue
            # Only check when next word looks like a modifier (not a verb)
            next_word = words[i + 1]
            if next_word.startswith(("دە", "بی", "نا", "نە", "مە")):
                continue
            final_char = base[-1]
            if final_char in _vowels and not word.endswith("یی"):
                violations.append(
                    f"Ezafe allomorph: vowel-final '{base}' should take یی "
                    f"not single ی (F#165)"
                )
        
        return violations
    
    def _check_tense_consistency(self, sentence: str) -> list[str]:
        """Check tense marker consistency within a clause.
        
        Rules enforced:
        - F#254: In و-coordinated clauses, non-past → past sequence is
          ungrammatical (Maaruf 2009, pp. 84-85). Only past→non-past and
          same-tense are valid.
        - Mixed tense prefixes (دە/ئە with past stem, or no prefix with
          present stem) within a single clause flag inconsistency.
        """
        violations = []
        words = self._analyzer.tokenize(sentence)
        
        # Split on و to find coordinated clauses, but only when و
        # appears between verb-bearing segments (not NP-internal و).
        # Improved: only split when the preceding segment contains a verb,
        # preventing false splits on NP-internal "و" (e.g., "نان و پەنیر").
        verb_prefixes = ("دە", "ئە", "نا", "بی", "بە", "ب")
        clauses: list[list[str]] = [[]]
        for word in words:
            if word == "و" and len(clauses[-1]) > 1:
                # Only split if current clause has verb evidence
                has_verb = any(
                    w.startswith(vp) for w in clauses[-1] for vp in verb_prefixes
                )
                if has_verb:
                    clauses.append([])
                else:
                    clauses[-1].append(word)
            else:
                clauses[-1].append(word)
        
        # Filter out clauses that contain no verb evidence (likely NP
        # fragments from NP-internal و splits).
        clauses = [
            c for c in clauses
            if any(w.startswith(vp) for w in c for vp in verb_prefixes)
        ] or clauses  # keep original if no verb clauses found
        
        # Determine tense of each clause
        clause_tenses: list[Optional[str]] = []
        for clause_words in clauses:
            tense = self._detect_clause_tense(clause_words)
            clause_tenses.append(tense)
        
        # F#254: Check sequential tense ordering
        for i in range(len(clause_tenses) - 1):
            t1 = clause_tenses[i]
            t2 = clause_tenses[i + 1]
            if t1 and t2:
                # Non-past followed by past is blocked
                if t1 == "present" and t2 == "past":
                    violations.append(
                        f"Tense sequencing violation: non-past clause followed by "
                        f"past clause in و-coordination (F#254)"
                    )
        
        return violations
    
    @staticmethod
    def _detect_clause_tense(words: list[str]) -> Optional[str]:
        """Detect the dominant tense of a clause from verb morphology.

        Delegates to builder helpers for present-tense detection; falls back
        to stem matching for past tense.
        """
        for word in words:
            if _is_present_verb(word):
                return "present"
            # نا + verb stem = negated present (e.g. ناکات, نازانم).
            # Guard against false positives on nouns like نانی by requiring
            # a present-tense verb ending after the prefix.
            if word.startswith(_NEGATION_PRESENT_PREFIX) and len(word) > 3:
                remainder = word[len(_NEGATION_PRESENT_PREFIX):]
                if any(remainder.endswith(s) for s in _PRESENT_ENDINGS):
                    return "present"
            if word.startswith(_IMPERATIVE_PREFIX) and len(word) > 1:
                return "present"
        # Past: check for known transitive/intransitive past stems
        for word in words:
            if _is_present_verb(word):
                continue
            for stem in TRANSITIVE_PAST_STEMS:
                if word.startswith(stem) or (
                    word.startswith(_NEGATION_PAST_PREFIX)
                    and word[len(_NEGATION_PAST_PREFIX):].startswith(stem)
                ):
                    return "past"
            if any(word.endswith(p) for p in ("مان", "تان", "یان")) and len(word) > 4:
                return "past"
        return None

    def _check_object_verb_ergative(self, sentence: str) -> list[str]:
        """Check object-verb agreement in past transitive clauses (Law 2).

        In Sorani Kurdish, past transitive verbs agree with the object in
        person and number — not the subject. This is the ergative split
        described by Slevanayi (2001, pp. 60-61, 89) and implemented in
        the builder as Step 3 (VS) and Step 7 (VO).

        Checks: if a past transitive verb carries a Set 2 suffix whose
        person/number conflicts with a nearby definite object NP, flag it.

        EVAL-5 fix: scans backward to clause boundary instead of a fixed
        6-word window, catching distant object-verb pairs within a clause.
        """
        violations = []
        words = self._analyzer.tokenize(sentence)
        clause_bounds = set(self._clause_boundary_indices(words))

        for i, word in enumerate(words):
            # Skip present-tense verbs
            if _is_present_verb(word):
                continue
            if not _is_transitive_past(word):
                continue

            # Extract verb person/number from suffix
            verb_pn = self._verb_ending_to_pn(word)
            if verb_pn is None:
                continue
            verb_person, verb_number = verb_pn

            # Scan backward to clause boundary for the nearest object
            for j in range(i - 1, -1, -1):
                if j in clause_bounds:
                    break
                obj = words[j]
                obj_person: str | None = None
                obj_number: str | None = None

                if obj in _PRONOUN_AGREEMENT:
                    obj_person, obj_number = _PRONOUN_AGREEMENT[obj]
                elif obj.endswith("ەکان") or obj.endswith("یەکان"):
                    obj_person, obj_number = "3", "pl"
                elif obj.endswith("ەکە") or obj.endswith("یەکە"):
                    obj_person, obj_number = "3", "sg"
                else:
                    continue

                if obj_person and (
                    obj_person != verb_person or obj_number != verb_number
                ):
                    violations.append(
                        f"Ergative mismatch (Law 2): object '{obj}' "
                        f"({obj_person}{obj_number}) with past transitive "
                        f"verb '{word}' ({verb_person}{verb_number})"
                    )
                break  # only check nearest object

        return violations

    # ── PIPE-9: Additional checks for uncovered error generators ──

    # Common orthographic confusion pairs (subset of what orthography.py generates)
    _ORTHO_CONFUSIONS = [
        ("ح", "ه"),
        ("خ", "غ"),
        ("ڵ", "ل"),
        ("ڕ", "ر"),
        ("ع", "ئ"),
    ]

    def _check_orthography(self, sentence: str) -> list[str]:
        """Flag words containing likely orthographic confusions.

        Uses the lexicon (when available) to check whether a word with a
        known confusion character is misspelled. This catches cases where
        the model preserved a misspelling instead of correcting it.
        """
        violations = []
        words = self._analyzer.tokenize(sentence)
        for word in words:
            for a, b in self._ORTHO_CONFUSIONS:
                if a in word:
                    alt = word.replace(a, b, 1)
                    if (self._analyzer._lexicon
                            and hasattr(self._analyzer._lexicon, 'is_correct')
                            and not self._analyzer._lexicon.is_correct(word)
                            and self._analyzer._lexicon.is_correct(alt)):
                        violations.append(
                            f"Orthographic confusion: '{word}' may be '{alt}' "
                            f"({a}→{b})"
                        )
                        break
        return violations

    _NEG_MARKERS = {"نە", "نا", "هیچ", "هەرگیز"}

    def _check_negative_concord(self, sentence: str) -> list[str]:
        """Check negative concord: هیچ/هەرگیز require نە/نا on the verb.

        In Sorani Kurdish, negative polarity items like هیچ (nothing) and
        هەرگیز (never) require a negated verb in the same clause. A
        sentence like *هیچ دەزانم is ungrammatical.
        """
        violations = []
        words = self._analyzer.tokenize(sentence)
        npi_words = {"هیچ", "هەرگیز", "هیچکەس", "هیچکام"}
        has_npi = any(w in npi_words for w in words)
        if not has_npi:
            return violations
        has_neg_verb = any(
            w.startswith("نا") or w.startswith("نە") for w in words
        )
        if not has_neg_verb:
            npis = [w for w in words if w in npi_words]
            violations.append(
                f"Negative concord: NPI {npis} without negated verb"
            )
        return violations

    def _check_conditional_agreement(self, sentence: str) -> list[str]:
        """Check conditional clause tense constraints.

        ئەگەر (if) clauses in Sorani typically take subjunctive or past
        tense in the protasis, not indicative present with دە-prefix.
        A sentence like *ئەگەر دەڕۆم is non-standard; the correct form
        uses the bare subjunctive (ئەگەر بچم).
        """
        violations = []
        words = self._analyzer.tokenize(sentence)
        cond_markers = {"ئەگەر", "ئەگەری"}
        in_cond = False
        for i, word in enumerate(words):
            if word in cond_markers:
                in_cond = True
                continue
            if in_cond:
                # End condition at clause boundaries
                if word in {"،", ".", "؟", "!"}:
                    in_cond = False
                    continue
                # دە-prefix in conditional protasis is non-standard
                if word.startswith("دە") and _is_present_verb(word):
                    violations.append(
                        f"Conditional agreement: indicative '{word}' in "
                        f"ئەگەر-clause; expected subjunctive (ب-prefix)"
                    )
                    in_cond = False
        return violations

    # ── CRIT-4: Four additional agreement checks ──

    # Sorani quantifiers that force plural agreement on the verb
    # (Slevanayi 2001, pp. 87-88; Maaruf 2010, p. 139)
    _QUANTIFIERS_PLURAL = {"هەر", "هیچ", "هەموو", "چەند", "هەندێک"}

    def _check_quantifier_noun(self, sentence: str) -> list[str]:
        """Check quantifier–verb number agreement.

        In Sorani Kurdish, certain quantifiers (هەموو, هەر, هیچ, چەند,
        هەندێک) govern a plural verb. For example:
        هەموو منداڵ *دەچێت is wrong; the correct form is
        هەموو منداڵ دەچن (3pl).

        Source: Slevanayi (2001), pp. 87-88; Maaruf (2010), p. 139.
        """
        violations = []
        words = self._analyzer.tokenize(sentence)
        clause_bounds = set(self._clause_boundary_indices(words))

        for i, word in enumerate(words):
            if word not in self._QUANTIFIERS_PLURAL:
                continue
            # Scan forward for a verb within the same clause
            for j in range(i + 1, len(words)):
                if j in clause_bounds:
                    break
                candidate = words[j]
                is_present = any(candidate.startswith(p) for p in _PRESENT_PREFIXES)
                is_neg_present = candidate.startswith(_NEGATION_PRESENT_PREFIX)
                if not (is_present or is_neg_present):
                    continue
                verb_pn = self._verb_ending_to_pn(candidate)
                if verb_pn is None:
                    break
                _, verb_number = verb_pn
                if verb_number == "sg":
                    violations.append(
                        f"Quantifier–verb mismatch: '{word}' requires plural "
                        f"verb, but '{candidate}' is singular"
                    )
                break
        return violations

    # Sorani relative clause markers
    _REL_MARKERS = {"کە", "ئەوەی"}

    def _check_relative_clause(self, sentence: str) -> list[str]:
        """Check antecedent–verb agreement in relative clauses.

        When a relative clause (introduced by کە or ئەوەی) modifies an
        antecedent, the verb inside the relative clause should agree in
        person and number with the antecedent head noun—not with any
        intervening NP.

        Heuristic: if the word before کە is a pronoun with known person/
        number, the first verb after کە should agree with it.
        """
        violations = []
        words = self._analyzer.tokenize(sentence)

        for i, word in enumerate(words):
            if word not in self._REL_MARKERS:
                continue
            # Antecedent is the previous word (head noun of the NP)
            if i == 0:
                continue
            antecedent = words[i - 1]
            ant_person: str | None = None
            ant_number: str | None = None
            if antecedent in _PRONOUN_AGREEMENT:
                ant_person, ant_number = _PRONOUN_AGREEMENT[antecedent]
            elif antecedent.endswith("ەکان") or antecedent.endswith("یەکان"):
                ant_person, ant_number = "3", "pl"
            elif antecedent.endswith("ەکە") or antecedent.endswith("یەکە"):
                ant_person, ant_number = "3", "sg"
            else:
                continue  # cannot determine antecedent features

            # Scan for the first verb inside the relative clause
            for j in range(i + 1, len(words)):
                if words[j] in {"،", ".", "؟", "!"}:
                    break
                candidate = words[j]
                is_present = any(candidate.startswith(p) for p in _PRESENT_PREFIXES)
                is_neg_present = candidate.startswith(_NEGATION_PRESENT_PREFIX)
                if not (is_present or is_neg_present):
                    continue
                verb_pn = self._verb_ending_to_pn(candidate)
                if verb_pn is None:
                    break
                verb_person, verb_number = verb_pn
                if ant_number and verb_number != ant_number:
                    violations.append(
                        f"Relative clause number mismatch: antecedent '{antecedent}' "
                        f"({ant_person}{ant_number}) but verb '{candidate}' "
                        f"({verb_person}{verb_number}) in کە-clause"
                    )
                if ant_person and verb_person != ant_person:
                    violations.append(
                        f"Relative clause person mismatch: antecedent '{antecedent}' "
                        f"({ant_person}{ant_number}) but verb '{candidate}' "
                        f"({verb_person}{verb_number}) in کە-clause"
                    )
                break
        return violations

    # Vocative markers and imperative detection
    _VOCATIVE_MARKERS = {"ئەی", "یا"}

    def _check_vocative_imperative(self, sentence: str) -> list[str]:
        """Check vocative marker–imperative verb number agreement.

        A sentence beginning with a vocative marker (ئەی, یا) followed
        by a singular addressee should have a 2sg imperative; with a
        plural addressee (or plural noun), 2pl imperative.

        Imperative verbs in Sorani start with ب- (or بی-).
        """
        violations = []
        words = self._analyzer.tokenize(sentence)
        if not words:
            return violations

        if words[0] not in self._VOCATIVE_MARKERS:
            return violations

        # Determine addressee number from the noun after vocative marker
        addressee_number: str | None = None
        for k in range(1, min(len(words), 4)):
            w = words[k]
            if w.endswith("ەکان") or w.endswith("یەکان"):
                addressee_number = "pl"
                break
            elif w.endswith("ەکە") or w.endswith("یەکە"):
                addressee_number = "sg"
                break
            elif w in _PRONOUN_AGREEMENT:
                _, addressee_number = _PRONOUN_AGREEMENT[w]
                break

        if addressee_number is None:
            return violations

        # Find imperative verb (ب-prefix)
        for word in words:
            if word.startswith("ب") and len(word) > 2 and not _is_present_verb(word):
                # Imperative 2sg typically ends without ن; 2pl ends with ن
                if addressee_number == "pl" and not word.endswith("ن"):
                    violations.append(
                        f"Vocative-imperative mismatch: plural addressee "
                        f"but imperative '{word}' is singular"
                    )
                elif addressee_number == "sg" and word.endswith("ن"):
                    violations.append(
                        f"Vocative-imperative mismatch: singular addressee "
                        f"but imperative '{word}' is plural"
                    )
                break
        return violations

    # Temporal adverbs with tense constraints
    _PAST_ADVERBS = {"دوێنێ", "پار", "پێشتر", "بەرلە", "پارێ"}
    _PRESENT_ADVERBS = {"ئێستا", "ئەمڕۆ", "دواتر", "هەمیشە"}

    def _check_adverb_verb_tense(self, sentence: str) -> list[str]:
        """Check temporal adverb–verb tense consistency.

        Temporal adverbs constrain the tense of the clause verb. A past
        adverb (دوێنێ = yesterday, پار = last year) with a present-tense
        verb is inconsistent, and vice versa.
        """
        violations = []
        words = self._analyzer.tokenize(sentence)
        has_past_adv = any(w in self._PAST_ADVERBS for w in words)
        has_present_adv = any(w in self._PRESENT_ADVERBS for w in words)

        if not (has_past_adv or has_present_adv):
            return violations

        clause_tense = self._detect_clause_tense(words)
        if clause_tense is None:
            return violations

        if has_past_adv and clause_tense == "present":
            adverbs = [w for w in words if w in self._PAST_ADVERBS]
            violations.append(
                f"Adverb-tense mismatch: past adverb(s) {adverbs} "
                f"with present-tense verb"
            )
        if has_present_adv and clause_tense == "past":
            adverbs = [w for w in words if w in self._PRESENT_ADVERBS]
            violations.append(
                f"Adverb-tense mismatch: present adverb(s) {adverbs} "
                f"with past-tense verb"
            )
        return violations

    # Person hierarchy for compound subjects: 1st > 2nd > 3rd
    _PERSON_HIERARCHY = {"1": 3, "2": 2, "3": 1}

    def _check_compound_subject(self, sentence: str) -> list[str]:
        """Check compound subject person resolution (Slevanayi 2001, p. 89).

        When two subjects are coordinated with و, the verb should agree with
        the highest person in the hierarchy: 1st > 2nd > 3rd.
        Example: من و تۆ دەچین (I and you go-1pl), NOT *من و تۆ دەچن (go-3pl).
        """
        violations = []
        words = self._analyzer.tokenize(sentence)

        # Find coordinated pronoun subjects: pronoun + و + pronoun
        pronouns_found = []
        for i, word in enumerate(words):
            if word in _PRONOUN_AGREEMENT:
                pronouns_found.append((i, word))

        if len(pronouns_found) < 2:
            return violations

        # Check if pronouns are coordinated with و
        coordinated_pronouns = []
        for idx in range(len(pronouns_found) - 1):
            pos_a = pronouns_found[idx][0]
            pos_b = pronouns_found[idx + 1][0]
            # Check for و between the two pronouns
            if pos_b - pos_a == 2 and pos_a + 1 < len(words) and words[pos_a + 1] == "و":
                coordinated_pronouns.append(pronouns_found[idx][1])
                coordinated_pronouns.append(pronouns_found[idx + 1][1])

        if len(coordinated_pronouns) < 2:
            return violations

        # Determine expected person: highest in hierarchy
        persons = [_PRONOUN_AGREEMENT[p][0] for p in coordinated_pronouns]
        expected_person = max(persons, key=lambda p: self._PERSON_HIERARCHY.get(p, 0))

        # Find the verb after the compound subject
        last_pronoun_pos = max(pos for pos, _ in pronouns_found if _ in coordinated_pronouns)
        for j in range(last_pronoun_pos + 1, min(last_pronoun_pos + 8, len(words))):
            word = words[j]
            for ending, (person, _number) in _PRESENT_ENDINGS.items():
                if word.endswith(ending) and any(word.startswith(p) for p in _PRESENT_PREFIXES):
                    if person != expected_person:
                        violations.append(
                            "Compound subject person mismatch: "
                            f"coordinated pronouns {coordinated_pronouns} "
                            f"expect person={expected_person} but verb "
                            f"'{word}' has person={person}"
                        )
                    return violations

        return violations

    # Common bare nouns (non-pronominal) — treated as 3sg for agreement
    _BARE_NOUN_INDICATORS = {"پیاو", "ژن", "منداڵ", "مامۆستا", "قوتابی", "کچ", "کوڕ"}

    def _check_bare_noun_agreement(self, sentence: str) -> list[str]:
        """Check bare noun person-only agreement (Slevanayi 2001, p. 60).

        Bare nouns (without demonstrative/definite marker) agree with the
        verb in person only (3sg) without number constraint. A 1st or 2nd
        person verb after a bare noun subject is a violation.
        """
        violations = []
        words = self._analyzer.tokenize(sentence)

        for i, word in enumerate(words):
            if word not in self._BARE_NOUN_INDICATORS:
                continue
            # Look ahead for a verb, skip if a demonstrative precedes
            if i > 0 and words[i - 1] in ("ئەو", "ئەم", "ئەوان"):
                continue

            for j in range(i + 1, min(i + 8, len(words))):
                v = words[j]
                if v == "و" or v in ("،", ".", "؟", "!"):
                    break
                for ending, (person, _number) in _PRESENT_ENDINGS.items():
                    if v.endswith(ending) and any(v.startswith(p) for p in _PRESENT_PREFIXES):
                        if person in ("1", "2"):
                            violations.append(
                                f"Bare noun '{word}' expects 3rd person verb "
                                f"but '{v}' has person={person}"
                            )
                        return violations

        return violations


def evaluate_agreement_accuracy(
    sentences: list[str],
    checker: Optional[AgreementChecker] = None,
) -> dict:
    """Compute agreement accuracy over a corpus.
    
    Returns:
        Dict with 'accuracy', 'total_sentences', 'correct_sentences', 'details'.
    """
    if checker is None:
        checker = AgreementChecker()
    
    results = [checker.check_sentence(s) for s in sentences]
    
    correct = sum(1 for r in results if r.is_correct)
    total = len(results)
    
    return {
        "accuracy": correct / total if total > 0 else 0.0,
        "total_sentences": total,
        "correct_sentences": correct,
        "avg_checks_passed": sum(r.checks_passed for r in results) / total if total > 0 else 0,
        "avg_checks_total": sum(r.checks_total for r in results) / total if total > 0 else 0,
    }


# PIPE-4: Per-agreement-law breakdown — labels match check_sentence order
_CHECK_LABELS: list[tuple[str, str]] = [
    ("subject_verb", "Law 1"),     # Check 1
    ("clitic_consistency", ""),     # Check 2
    ("ezafe", ""),                  # Check 3
    ("tense_consistency", ""),      # Check 4
    ("object_verb_ergative", "Law 2"),  # Check 5
    ("negative_concord", ""),       # Check 6
    ("orthography", ""),            # Check 7
    ("conditional", ""),            # Check 8
    ("quantifier_noun", ""),        # Check 9
    ("relative_clause", ""),        # Check 10
    ("vocative_imperative", ""),    # Check 11
    ("adverb_verb_tense", ""),      # Check 12
    ("compound_subject", ""),       # Check 13
    ("bare_noun", ""),              # Check 14
]


def evaluate_agreement_by_check(
    sentences: list[str],
    checker: Optional[AgreementChecker] = None,
) -> dict[str, dict]:
    """Per-check accuracy breakdown, returning stats for each of the 14 checks.

    Also aggregates Law 1 (subject-verb) and Law 2 (object-verb ergative)
    separately — the two agreement laws central to this thesis.
    """
    if checker is None:
        checker = AgreementChecker()

    per_check: dict[str, dict] = {}
    for label, _law in _CHECK_LABELS:
        per_check[label] = {"correct": 0, "total": 0, "law": _law}

    check_methods = [
        "_check_subject_verb",
        "_check_clitic_consistency",
        "_check_ezafe",
        "_check_tense_consistency",
        "_check_object_verb_ergative",
        "_check_negative_concord",
        "_check_orthography",
        "_check_conditional_agreement",
        "_check_quantifier_noun",
        "_check_relative_clause",
        "_check_vocative_imperative",
        "_check_adverb_verb_tense",
        "_check_compound_subject",
        "_check_bare_noun_agreement",
    ]

    for sent in sentences:
        for (label, _law), method_name in zip(_CHECK_LABELS, check_methods):
            method = getattr(checker, method_name)
            violations = method(sent)
            per_check[label]["total"] += 1
            if not violations:
                per_check[label]["correct"] += 1

    # Compute accuracy per check
    for label in per_check:
        t = per_check[label]["total"]
        c = per_check[label]["correct"]
        per_check[label]["accuracy"] = c / t if t > 0 else 0.0

    # Aggregate Law 1 / Law 2
    law_summary = {}
    for label, info in per_check.items():
        law = info.get("law", "")
        if law:
            if law not in law_summary:
                law_summary[law] = {"correct": 0, "total": 0}
            law_summary[law]["correct"] += info["correct"]
            law_summary[law]["total"] += info["total"]
    for law in law_summary:
        t = law_summary[law]["total"]
        c = law_summary[law]["correct"]
        law_summary[law]["accuracy"] = c / t if t > 0 else 0.0

    return {"per_check": per_check, "per_law": law_summary}
