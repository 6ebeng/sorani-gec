import logging
from dataclasses import dataclass, field
from typing import Optional
from ...morphology.analyzer import (
    MorphologicalAnalyzer,
    CLITIC_PERSON_MAP,
)
from ...morphology.constants import (
    SUBJECT_PRONOUNS,
    TRANSITIVE_PAST_STEMS,
    CLITIC_BARRED_PRONOUNS,
    RECIPROCAL_VARIANTS,
)
from ...morphology.builder import (
    _is_present_verb,
    _is_transitive_past,
    _is_past_verb,
    build_agreement_graph,
)

from typing import Optional
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

from .constants import *
from .simplesentence import SimpleSentenceMixin
from .compoundsentence import CompoundSentenceMixin
from .complexsentence import ComplexSentenceMixin

@dataclass
class AgreementResult:
    """Result of agreement checking on a single sentence.

    ``checks_total`` is the number of checks run (always 15). ``checks_applicable``
    is how many of those checks had their trigger structure present in the
    sentence — the denominator that matters. A check that never fired (no
    pronoun, no quantifier, no relative clause, …) is neither a pass nor a
    fail; counting it as a silent pass is what inflated the old headline
    toward 1.0. ``accuracy`` divides passed checks by *applicable* checks.
    """
    sentence: str
    checks_passed: int
    checks_total: int
    violations: list[str]
    checks_applicable: int = 0
    simple_violations: list[str] = field(default_factory=list)
    compound_violations: list[str] = field(default_factory=list)
    complex_violations: list[str] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        # Denominator = applicable checks, not all 15. A sentence that
        # triggered no check is undefined for this metric; callers should
        # gate on is_applicable. To keep the property total we return 1.0
        # when nothing applied.
        if self.checks_applicable <= 0:
            return 1.0
        failed = self.checks_total - self.checks_passed
        passed_applicable = self.checks_applicable - failed
        return passed_applicable / self.checks_applicable

    @property
    def is_applicable(self) -> bool:
        return self.checks_applicable > 0

    @property
    def is_correct(self) -> bool:
        return len(self.violations) == 0


class AgreementChecker(SimpleSentenceMixin, CompoundSentenceMixin, ComplexSentenceMixin):
    def __init__(self, analyzer: Optional[MorphologicalAnalyzer] = None):
        self._analyzer = analyzer or MorphologicalAnalyzer(use_klpt=False)

    def check_sentence(self, sentence: str) -> AgreementResult:
        """Run all agreement checks on a sentence.
    
        Each check returns ``(applicable, violations)``. ``applicable`` is
        True only when the check's trigger structure is present (a pronoun
        for subject-verb, a quantifier for quantifier-noun, …). Checks that
        do not apply are excluded from the denominator instead of silently
        counting as passes.
        """
        violations: list[str] = []
        failed_checks = 0
        applicable_checks = 0
        total_checks = 0
    
        # Modularized agreements by sentence type
        simple_sentence_checks = [
            self._check_subject_verb,
            self._check_clitic_consistency,
            self._check_ezafe,
            self._check_object_verb_ergative,
            self._check_negative_concord,
            self._check_orthography,
            self._check_quantifier_noun,
            self._check_vocative_imperative,
            self._check_adverb_verb_tense,
            self._check_bare_noun_agreement,
            self._check_noun_subject_verb_number,
        ]
        
        compound_sentence_checks = [
            self._check_compound_subject,
            self._check_tense_consistency,
            self._check_cross_clause_covert_subject,
        ]
        
        complex_sentence_checks = [
            self._check_conditional_agreement,
            self._check_relative_clause,
        ]
    
        simple_violations = []
        for check in simple_sentence_checks:
            applicable, check_violations = check(sentence)
            simple_violations.extend(check_violations)
            total_checks += 1
            if applicable:
                applicable_checks += 1
            if check_violations:
                failed_checks += 1
    
        compound_violations = []
        for check in compound_sentence_checks:
            applicable, check_violations = check(sentence)
            compound_violations.extend(check_violations)
            total_checks += 1
            if applicable:
                applicable_checks += 1
            if check_violations:
                failed_checks += 1
    
        complex_violations = []
        for check in complex_sentence_checks:
            applicable, check_violations = check(sentence)
            complex_violations.extend(check_violations)
            total_checks += 1
            if applicable:
                applicable_checks += 1
            if check_violations:
                failed_checks += 1
    
        passed = total_checks - failed_checks
        violations = simple_violations + compound_violations + complex_violations
    
        return AgreementResult(
            sentence=sentence,
            checks_passed=passed,
            checks_total=total_checks,
            violations=violations,
            checks_applicable=applicable_checks,
            simple_violations=simple_violations,
            compound_violations=compound_violations,
            complex_violations=complex_violations,
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

    @staticmethod
    def _verb_ending_to_pn(verb: str) -> Optional[tuple[str, str]]:
        """Extract person/number from a verb's ending suffix."""
        # Check longest suffixes first
        for suffix in sorted(_PRESENT_ENDINGS, key=len, reverse=True):
            if verb.endswith(suffix):
                return _PRESENT_ENDINGS[suffix]
        return None

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


def evaluate_agreement_accuracy(
    sentences: list[str],
    checker: Optional[AgreementChecker] = None,
) -> dict:
    """Compute agreement accuracy over a corpus.

    Reports two denominators. ``accuracy`` keeps the legacy sentence-level
    pass rate (fraction of sentences with no violation), which is inflated
    on corpora full of sentences where no check applies. ``accuracy_applicable``
    restricts to sentences with at least one applicable check ΓÇö the
    denominator the metric was meant to use.

    Returns:
        Dict with accuracy fields, totals, and average check counts.
    """
    if checker is None:
        checker = AgreementChecker()

    results = [checker.check_sentence(s) for s in sentences]

    correct = sum(1 for r in results if r.is_correct)
    total = len(results)

    applicable_results = [r for r in results if r.is_applicable]
    n_applicable = len(applicable_results)
    correct_applicable = sum(1 for r in applicable_results if r.is_correct)

    return {
        "accuracy": correct / total if total > 0 else 0.0,
        "accuracy_applicable": (
            correct_applicable / n_applicable if n_applicable > 0 else 0.0
        ),
        "total_sentences": total,
        "correct_sentences": correct,
        "applicable_sentences": n_applicable,
        "correct_applicable_sentences": correct_applicable,
        "avg_checks_passed": sum(r.checks_passed for r in results) / total if total > 0 else 0,
        "avg_checks_total": sum(r.checks_total for r in results) / total if total > 0 else 0,
        "avg_checks_applicable": (
            sum(r.checks_applicable for r in results) / total if total > 0 else 0
        ),
    }

def evaluate_agreement_by_check(
    sentences: list[str],
    checker: Optional[AgreementChecker] = None,
) -> dict[str, dict]:
    """Per-check accuracy breakdown, returning stats for each of the 15 checks.

    Also aggregates Law 1 (subject-verb) and Law 2 (object-verb ergative)
    separately ΓÇö the two agreement laws central to this thesis.
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
        "_check_noun_subject_verb_number",
    ]

    for sent in sentences:
        for (label, _law), method_name in zip(_CHECK_LABELS, check_methods):
            method = getattr(checker, method_name)
            applicable, violations = method(sent)
            # Only count a check toward its denominator when it applied ΓÇö
            # inapplicable checks are not silent passes.
            if not applicable:
                continue
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
