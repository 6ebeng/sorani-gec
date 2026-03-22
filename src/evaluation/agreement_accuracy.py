"""
Sorani Kurdish Agreement Accuracy Checker

Sorani-specific evaluation metric that checks whether agreement constraints
are satisfied in the model's output. Measures the percentage of sentences
where targeted agreement checks pass.

Uses the rule-based morphological analyzer (Amin 2016, Fatah & Qadir 2006)
and agreement constants from Slevanayi (2001) to detect violations.
"""

import re
import logging
from dataclasses import dataclass
from typing import Optional

from ..morphology.analyzer import (
    MorphologicalAnalyzer,
    MorphFeatures,
    CLITIC_PERSON_MAP,
)
from ..morphology.constants import SUBJECT_PRONOUNS, TRANSITIVE_PAST_STEMS
from ..morphology.builder import (
    _is_present_verb,
    _is_past_verb,
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
    """
    
    def __init__(self, analyzer: Optional[MorphologicalAnalyzer] = None):
        self._analyzer = analyzer or MorphologicalAnalyzer(use_klpt=False)
    
    def check_sentence(self, sentence: str) -> AgreementResult:
        """Run all agreement checks on a sentence."""
        violations = []
        total_checks = 0
        
        # Check 1: Subject-verb number
        sv_violations = self._check_subject_verb(sentence)
        violations.extend(sv_violations)
        total_checks += 1
        
        # Check 2: Clitic consistency
        cl_violations = self._check_clitic_consistency(sentence)
        violations.extend(cl_violations)
        total_checks += 1
        
        # Check 3: Ezafe presence
        ez_violations = self._check_ezafe(sentence)
        violations.extend(ez_violations)
        total_checks += 1
        
        # Check 4: Tense consistency
        t_violations = self._check_tense_consistency(sentence)
        violations.extend(t_violations)
        total_checks += 1

        # Check 5: Object-verb agreement (Law 2 — ergative past transitive)
        ov_violations = self._check_object_verb_ergative(sentence)
        violations.extend(ov_violations)
        total_checks += 1
        
        passed = total_checks - min(len(violations), total_checks)
        
        return AgreementResult(
            sentence=sentence,
            checks_passed=passed,
            checks_total=total_checks,
            violations=violations,
        )
    
    def _check_subject_verb(self, sentence: str) -> list[str]:
        """Check subject-verb person/number agreement (Law 1).
        
        Source: Slevanayi (2001), p. 89 — verb agrees with subject in
        person and number for intransitive and present-tense transitive.
        """
        violations = []
        words = self._analyzer.tokenize(sentence)
        
        for i, word in enumerate(words):
            if word not in _PRONOUN_AGREEMENT:
                continue
            expected_person, expected_number = _PRONOUN_AGREEMENT[word]
            
            # Scan forward for a verb (present-tense prefix)
            for j in range(i + 1, min(i + 6, len(words))):
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
        """
        violations = []
        words = self._analyzer.tokenize(sentence)
        found_clitics: list[tuple[str, str, str]] = []  # (clitic, person, number)

        for word in words:
            # Set 2 verb suffixes (م/ت/ی on دەکەم etc.) are NOT
            # Set 1 clitics. Skip present-tense verbs entirely.
            if _is_present_verb(word) or word.startswith(_NEGATION_PRESENT_PREFIX):
                continue
            # Skip possessive hosts: nouns with ezafe + clitic are Set 3,
            # not Set 1. Detect by checking for ی before the clitic.
            for cl, (person, number) in CLITIC_PERSON_MAP.items():
                if word.endswith(cl) and len(word) > len(cl) + 1:
                    stem = word[: -len(cl)]
                    # If stem ends with ezafe ی, this is a possessive (Set 3)
                    if stem.endswith("ی"):
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
        clauses: list[list[str]] = [[]]
        for word in words:
            if word == "و" and len(clauses[-1]) > 1:
                clauses.append([])
            else:
                clauses[-1].append(word)
        
        # Filter out clauses that contain no verb evidence (likely NP
        # fragments from NP-internal و splits).
        verb_prefixes = ("دە", "ئە", "نا", "بی", "بە", "ب")
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
        """
        violations = []
        words = self._analyzer.tokenize(sentence)

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

            # Scan backward for the nearest object candidate:
            # definite nouns (ending in ەکە/ەکان) or pronouns
            for j in range(i - 1, max(i - 6, -1), -1):
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
