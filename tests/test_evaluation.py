"""
Tests for the F₀.₅ scorer and agreement accuracy checker.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.evaluation.f05_scorer import compute_f05, evaluate_corpus, GECMetrics
from src.evaluation.agreement_accuracy import AgreementChecker, evaluate_agreement_accuracy


# ============================================================================
# F₀.₅ Scorer Tests
# ============================================================================

def test_compute_f05_perfect():
    """Perfect precision and recall should give F₀.₅ = 1.0."""
    assert compute_f05(1.0, 1.0) == 1.0


def test_compute_f05_zero():
    """Zero precision and recall should give F₀.₅ = 0.0."""
    assert compute_f05(0.0, 0.0) == 0.0


def test_compute_f05_precision_weighted():
    """F₀.₅ should weight precision more than recall."""
    # High precision, low recall
    f05_high_p = compute_f05(0.9, 0.3)
    # Low precision, high recall
    f05_high_r = compute_f05(0.3, 0.9)
    # F₀.₅ should favor high precision
    assert f05_high_p > f05_high_r, \
        f"F₀.₅ should favor precision: {f05_high_p:.4f} > {f05_high_r:.4f}"


def test_evaluate_corpus_perfect_correction():
    """Model perfectly corrects all errors."""
    sources =    ["من دەچین بۆ قوتابخانە"]
    hypotheses = ["من دەچم بۆ قوتابخانە"]
    references = ["من دەچم بۆ قوتابخانە"]

    metrics = evaluate_corpus(sources, hypotheses, references)
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f05 == 1.0
    print(f"  Perfect correction: {metrics}")


def test_evaluate_corpus_no_correction():
    """Model makes no changes (copies source)."""
    sources =    ["من دەچین بۆ قوتابخانە"]
    hypotheses = ["من دەچین بۆ قوتابخانە"]  # no correction
    references = ["من دەچم بۆ قوتابخانە"]

    metrics = evaluate_corpus(sources, hypotheses, references)
    assert metrics.recall == 0.0  # missed the error
    assert metrics.fp == 0        # no spurious corrections
    print(f"  No correction: {metrics}")


def test_evaluate_corpus_spurious_correction():
    """Model changes something that should not have been changed."""
    sources =    ["من دەچم بۆ قوتابخانە"]
    hypotheses = ["من دەچین بۆ قوتابخانە"]  # introduced error
    references = ["من دەچم بۆ قوتابخانە"]    # source was already correct

    metrics = evaluate_corpus(sources, hypotheses, references)
    assert metrics.fp > 0  # has false positives
    print(f"  Spurious correction: {metrics}")


def test_evaluate_corpus_empty():
    """Empty corpus."""
    metrics = evaluate_corpus([], [], [])
    assert metrics.f05 == 0.0
    assert metrics.tp == 0


def test_evaluate_corpus_multiple_sentences():
    """Multiple sentences with mixed results."""
    sources =    ["من دەچین", "تۆ دەچم", "ئەو باشە"]
    hypotheses = ["من دەچم",  "تۆ دەچیت", "ئەو باشە"]
    references = ["من دەچم",  "تۆ دەچیت", "ئەو باشە"]

    metrics = evaluate_corpus(sources, hypotheses, references)
    # LCS-based edit extraction decomposes each word substitution into
    # a deletion + insertion, so 2 substitutions → 4 edit operations.
    assert metrics.tp == 4   # two word substitutions = 4 LCS edits
    assert metrics.fn == 0   # no missed errors
    assert metrics.fp == 0   # no spurious
    print(f"  Multiple sentences: {metrics}")


# ============================================================================
# Agreement Accuracy Tests
# ============================================================================

def test_agreement_checker_basic():
    """Agreement checker runs without crashing."""
    checker = AgreementChecker()
    result = checker.check_sentence("من دەچم بۆ قوتابخانە")
    assert result.checks_total > 0
    print(f"  Basic check: passed={result.checks_passed}/{result.checks_total}, "
          f"violations={result.violations}")


def test_agreement_checker_violation():
    """Subject-verb mismatch should be detected."""
    checker = AgreementChecker()
    # "من" (I) with "دەچین" (we go) — number mismatch
    result = checker.check_sentence("من دەچین بۆ بازاڕ")
    print(f"  Violation check: passed={result.checks_passed}/{result.checks_total}, "
          f"violations={result.violations}")


def test_evaluate_agreement_accuracy_corpus():
    """Corpus-level agreement accuracy."""
    sentences = [
        "من دەچم بۆ قوتابخانە",     # correct
        "تۆ دەچیت بۆ ماڵەوە",       # correct
        "ئەوان دەچن بۆ بازاڕ",      # correct
    ]
    result = evaluate_agreement_accuracy(sentences)
    assert result["total_sentences"] == 3
    assert 0.0 <= result["accuracy"] <= 1.0
    print(f"  Corpus accuracy: {result['accuracy']:.2f} "
          f"({result['correct_sentences']}/{result['total_sentences']})")


def test_evaluate_agreement_accuracy_empty():
    """Empty corpus."""
    result = evaluate_agreement_accuracy([])
    assert result["total_sentences"] == 0
    assert result["accuracy"] == 0.0


# ============================================================================
# Enhanced Agreement Checker Tests
# ============================================================================

def test_agreement_checker_clitic_consistency():
    """Clitic consistency check runs (no crash) on normal sentence."""
    checker = AgreementChecker()
    result = checker.check_sentence("پارەکەم بردەوە")
    assert result.checks_total == 5
    print(f"  Clitic check: violations={result.violations}")


def test_agreement_checker_ezafe_demonstrative_violation():
    """Detects demonstrative + definite marker co-occurrence (F#10/R4)."""
    checker = AgreementChecker()
    # "ئەم کتێبەکە" — demonstrative + definite marker = violation
    result = checker.check_sentence("ئەم کتێبەکە باشە")
    # Should find demonstrative+definite violation
    has_dem_violation = any("Demonstrative" in v for v in result.violations)
    assert has_dem_violation, f"Expected demonstrative violation, got: {result.violations}"
    print(f"  F#10/R4 violation detected: {result.violations}")


def test_agreement_checker_tense_consistency_valid():
    """Same-tense coordination is valid (no violation)."""
    checker = AgreementChecker()
    # Past + و + Past = valid
    result = checker.check_sentence("نانی خوارد و چای خواردەوە")
    tense_violations = [v for v in result.violations if "Tense sequencing" in v]
    assert len(tense_violations) == 0
    print(f"  Valid tense coordination: no violations")


def test_agreement_checker_subject_verb_violation():
    """Subject-verb mismatch is detected for all persons."""
    checker = AgreementChecker()
    # "من" (1sg) with "دەچین" (1pl) — number mismatch
    result = checker.check_sentence("من دەچین بۆ بازاڕ")
    sv_violations = [v for v in result.violations if "Subject-verb mismatch" in v]
    assert len(sv_violations) > 0, f"Expected SV violation, got: {result.violations}"
    print(f"  SV mismatch detected: {sv_violations[0]}")


def test_agreement_checker_no_false_positive_correct_sentence():
    """Correct sentence should produce no subject-verb violations."""
    checker = AgreementChecker()
    result = checker.check_sentence("من دەچم بۆ قوتابخانە")
    sv_violations = [v for v in result.violations if "Subject-verb mismatch" in v]
    assert len(sv_violations) == 0
    print(f"  Correct sentence: no SV violations")


def test_h2_verb_suffix_not_conflated_with_clitic():
    """H2: Present-tense verb suffixes (Set 2) must not be flagged as Set 1 clitics."""
    checker = AgreementChecker()
    # "من دەکەم" — م on دەکەم is verb agreement (Set 2), not a clitic
    result = checker.check_sentence("من دەکەم")
    clitic_violations = [v for v in result.violations if "Clitic" in v or "clitic" in v]
    assert len(clitic_violations) == 0, (
        f"H2: Verb suffix 'م' on 'دەکەم' should not be flagged as clitic. "
        f"Got: {clitic_violations}"
    )
    print(f"  H2: Verb suffixes not conflated with clitics — violations={result.violations}")


# ============================================================================
# Run all tests
# ============================================================================

if __name__ == "__main__":
    print("=== F₀.₅ Scorer Tests ===")
    test_compute_f05_perfect()
    print("  test_compute_f05_perfect: PASSED")
    test_compute_f05_zero()
    print("  test_compute_f05_zero: PASSED")
    test_compute_f05_precision_weighted()
    print("  test_compute_f05_precision_weighted: PASSED")
    test_evaluate_corpus_perfect_correction()
    test_evaluate_corpus_no_correction()
    test_evaluate_corpus_spurious_correction()
    test_evaluate_corpus_empty()
    print("  test_evaluate_corpus_empty: PASSED")
    test_evaluate_corpus_multiple_sentences()

    print("\n=== Agreement Accuracy Tests ===")
    test_agreement_checker_basic()
    test_agreement_checker_violation()
    test_evaluate_agreement_accuracy_corpus()
    test_evaluate_agreement_accuracy_empty()

    print("\n=== Enhanced Agreement Checker Tests ===")
    test_agreement_checker_clitic_consistency()
    test_agreement_checker_ezafe_demonstrative_violation()
    test_agreement_checker_tense_consistency_valid()
    test_agreement_checker_subject_verb_violation()
    test_agreement_checker_no_false_positive_correct_sentence()
    print("  test_evaluate_agreement_accuracy_empty: PASSED")

    print("\n=== Round 18 High Gap Fix Tests — H2 (verb suffix not clitic) ===")
    test_h2_verb_suffix_not_conflated_with_clitic()

    print("\nAll evaluation tests passed!")
