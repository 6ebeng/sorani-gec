"""
Tests for the F₀.₅ scorer and agreement accuracy checker.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.evaluation.f05_scorer import (
    compute_f05,
    evaluate_corpus,
    evaluate_corpus_span,
    evaluate_corpus_with_sentences,
    evaluate_sentence,
    span_based_edits,
)
from src.evaluation.agreement_accuracy import AgreementChecker, evaluate_agreement_accuracy, evaluate_agreement_by_check


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
    # With substitution detection, each word substitution is a single edit
    # (not decomposed into deletion + insertion).
    assert metrics.tp == 2   # two word substitutions
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
    assert result.checks_total == 15
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
# Span-based Edit Tests (EVAL-1)
# ============================================================================

def test_span_based_edits_substitution():
    """Span-based edits detect word substitution with position."""
    edits = span_based_edits("من دەچین بۆ بازاڕ", "من دەچم بۆ بازاڕ")
    assert len(edits) == 1
    assert edits[0].src_text == "دەچین"
    assert edits[0].tgt_text == "دەچم"
    assert edits[0].src_start == 1
    assert edits[0].edit_type == "morphological"  # shares stem دەچ
    print(f"  Span edit: {edits[0]}")


def test_span_based_edits_insertion():
    """Span-based edits detect word insertion."""
    edits = span_based_edits("من دەچم", "من دەچم بۆ بازاڕ")
    insertions = [e for e in edits if e.edit_type == "insertion"]
    assert len(insertions) >= 1
    print(f"  Insertions: {insertions}")


def test_span_based_edits_deletion():
    """Span-based edits detect word deletion."""
    edits = span_based_edits("من دەچم بۆ بازاڕ", "من دەچم")
    deletions = [e for e in edits if e.edit_type == "deletion"]
    assert len(deletions) >= 1
    print(f"  Deletions: {deletions}")


def test_span_based_edits_morphological_classification():
    """Morphological edits (shared stem >=50%) classified correctly."""
    edits = span_based_edits("دەچین", "دەچم")
    assert len(edits) == 1
    assert edits[0].edit_type == "morphological"


def test_span_based_edits_full_substitution():
    """Unrelated words classified as substitution (not morphological)."""
    edits = span_based_edits("کتێب", "قوتابخانە")
    assert len(edits) == 1
    assert edits[0].edit_type == "substitution"


def test_evaluate_corpus_span_basic():
    """Span-based corpus evaluation returns overall + per-type metrics."""
    sources =    ["من دەچین بۆ قوتابخانە"]
    hypotheses = ["من دەچم بۆ قوتابخانە"]
    references = ["من دەچم بۆ قوتابخانە"]

    overall, per_type = evaluate_corpus_span(sources, hypotheses, references)
    assert overall.f05 == 1.0
    assert "morphological" in per_type
    print(f"  Span eval: {overall}, types: {list(per_type.keys())}")


def test_evaluate_corpus_span_empty():
    """Empty corpus span evaluation."""
    overall, per_type = evaluate_corpus_span([], [], [])
    assert overall.f05 == 0.0
    assert len(per_type) == 0


# ============================================================================
# Sentence-level Metric Tests (EVAL-3)
# ============================================================================

def test_evaluate_sentence_perfect():
    """Single sentence, perfect correction."""
    m = evaluate_sentence(
        "من دەچین بۆ قوتابخانە",
        "من دەچم بۆ قوتابخانە",
        "من دەچم بۆ قوتابخانە",
    )
    assert m.f05 == 1.0
    assert m.tp == 1
    print(f"  Sentence perfect: {m}")


def test_evaluate_sentence_no_correction():
    """Sentence with no correction (source copied)."""
    m = evaluate_sentence(
        "من دەچین بۆ قوتابخانە",
        "من دەچین بۆ قوتابخانە",
        "من دەچم بۆ قوتابخانە",
    )
    assert m.recall == 0.0
    assert m.fn >= 1
    print(f"  Sentence no correction: {m}")


def test_evaluate_corpus_with_sentences_basic():
    """Corpus + sentence-level metrics returned together."""
    sources =    ["من دەچین", "تۆ دەچم"]
    hypotheses = ["من دەچم",  "تۆ دەچیت"]
    references = ["من دەچم",  "تۆ دەچیت"]

    corpus, sentences = evaluate_corpus_with_sentences(
        sources, hypotheses, references,
    )
    assert len(sentences) == 2
    assert sentences[0].f05 == 1.0
    assert sentences[1].f05 == 1.0
    assert corpus.f05 == 1.0
    print(f"  Corpus+sentences: corpus={corpus}")


def test_evaluate_corpus_with_sentences_mixed():
    """Mixed results: one perfect, one clean (no edits needed)."""
    sources =    ["من دەچین", "ئەو باشە"]
    hypotheses = ["من دەچم",  "ئەو باشە"]
    references = ["من دەچم",  "ئەو باشە"]

    corpus, sentences = evaluate_corpus_with_sentences(
        sources, hypotheses, references,
    )
    assert len(sentences) == 2
    assert sentences[0].f05 == 1.0
    assert sentences[1].tp == 0
    print(f"  Mixed: corpus={corpus}")


def test_evaluate_corpus_with_sentences_empty():
    """Empty corpus."""
    corpus, sentences = evaluate_corpus_with_sentences([], [], [])
    assert corpus.f05 == 0.0
    assert len(sentences) == 0


def test_agreement_checker_object_verb_ergative():
    """Object-verb ergative check (Law 2) runs and produces results."""
    checker = AgreementChecker()
    # Past transitive: object should agree with verb
    result = checker.check_sentence("کتێبەکە بردم")
    assert result.checks_total == 15, (
        f"Expected 15 checks (including ergative + new checks), got {result.checks_total}"
    )


# ============================================================================
# Manual F₀.₅ Verification Tests (Fix 5.6)
# ============================================================================

def test_compute_f05_known_values():
    """F₀.₅ with P=0.8, R=0.6 should be ≈ 0.7692.

    Manual: F₀.₅ = (1 + 0.25) * (0.8 * 0.6) / (0.25 * 0.8 + 0.6)
                  = 1.25 * 0.48 / (0.2 + 0.6)
                  = 0.6 / 0.8
                  = 0.75
    """
    result = compute_f05(0.8, 0.6)
    assert abs(result - 0.75) < 1e-6, f"Expected ~0.75, got {result}"
    print(f"  F₀.₅(P=0.8, R=0.6) = {result:.6f}")


def test_compute_f05_high_precision_low_recall():
    """F₀.₅ at P=1.0, R=0.2 — precision dominates.

    Manual: F₀.₅ = 1.25 * (1.0 * 0.2) / (0.25 * 1.0 + 0.2)
                  = 0.25 / 0.45
                  ≈ 0.5556
    """
    result = compute_f05(1.0, 0.2)
    assert abs(result - 0.25 / 0.45) < 1e-6, f"Expected ~0.5556, got {result}"
    print(f"  F₀.₅(P=1.0, R=0.2) = {result:.6f}")


def test_compute_f05_balanced():
    """F₀.₅ at P=0.5, R=0.5 should equal 0.5.

    Manual: F₀.₅ = 1.25 * 0.25 / (0.125 + 0.5) = 0.3125 / 0.625 = 0.5
    """
    result = compute_f05(0.5, 0.5)
    assert abs(result - 0.5) < 1e-6, f"Expected 0.5, got {result}"
    print(f"  F₀.₅(P=0.5, R=0.5) = {result:.6f}")


def test_agreement_checker_fourteen_checks_counted():
    """AgreementChecker must run exactly 15 checks per sentence."""
    checker = AgreementChecker()
    result = checker.check_sentence("من نانم خوارد")
    assert result.checks_total == 15
    print(f"  Fifteen checks confirmed: {result.checks_total}")


# ============================================================================
# Findings-based robustness rules (F#128, F#124, F#49, F#88, F#26)
# ============================================================================

def test_reciprocal_requires_plural_subject():
    """F#128: singular subject + یەکتر is flagged; plural subject passes."""
    checker = AgreementChecker()
    bad = checker.check_sentence("من یەکترم بینی")
    assert any("Reciprocal" in v for v in bad.violations), bad.violations
    good = checker.check_sentence("ئەوان یەکتریان بینی")
    assert not any("Reciprocal" in v for v in good.violations), good.violations
    print("  Reciprocal plural-subject rule (F#128) enforced")


def test_clitic_barred_pronoun_hi():
    """F#124: *هیم/هیمە flagged; هی + independent pronoun passes."""
    checker = AgreementChecker()
    bad = checker.check_sentence("ئەو کتێبە هیمە")
    assert any("Clitic-barred" in v for v in bad.violations), bad.violations
    good = checker.check_sentence("ئەو کتێبە هی منە")
    assert not any("Clitic-barred" in v for v in good.violations), good.violations
    print("  Clitic-barred pronoun rule (F#124) enforced")


def test_pronoun_ezafe_rules():
    """F#49/R20: pronoun+ە before modifier flagged; pronoun+ی never flagged."""
    checker = AgreementChecker()
    bad = checker.check_sentence("تۆە باش")
    assert any("Pronoun ezafe" in v for v in bad.violations), bad.violations
    good = checker.check_sentence("تۆی باش")
    assert not any("Ezafe allomorph" in v for v in good.violations), good.violations
    assert not any("Pronoun ezafe" in v for v in good.violations), good.violations
    print("  Pronoun ezafe rules (F#49/R20) enforced")


def test_compound_noun_subject_plural_verb():
    """F#88: N و N + singular intransitive verb flagged; plural passes."""
    checker = AgreementChecker()
    bad = checker.check_sentence("کچ و کوڕ هات")
    assert any("Compound noun subject" in v for v in bad.violations), bad.violations
    good = checker.check_sentence("کچ و کوڕ هاتن")
    assert not any("Compound noun subject" in v for v in good.violations), good.violations
    print("  Compound noun subject plural rule (F#88) enforced")


def test_cross_clause_covert_subject_consistency():
    """F#22: covert-subject markers must agree across و-coordinated clauses."""
    checker = AgreementChecker()
    bad = checker.check_sentence("چووین بۆ هەولێر و سەردانی خزمانم کرد")
    assert any("Cross-clause" in v for v in bad.violations), bad.violations
    for good_s in ("چووم بۆ هەولێر و سەردانی خزمانم کرد",
                   "چووین بۆ هەولێر و سەردانی خزمانمان کرد"):
        good = checker.check_sentence(good_s)
        assert not any("Cross-clause" in v for v in good.violations), (
            good_s, good.violations)
    print("  Cross-clause covert-subject rule (F#22) enforced")


def test_category_rules_f60_f127_f122():
    """F#60 allomorph, F#127 ەوە clitic position, F#122 ش/یش order."""
    checker = AgreementChecker()
    # F#60: consonant-final stem must take ەکە/ێک, not یەکە/یەک
    assert any("Determiner allomorph" in v for v in
               checker.check_sentence("کتێبیەکە باشە").violations)
    assert not any("Determiner allomorph" in v for v in
                   checker.check_sentence("قوتابییەکە باشە").violations)
    # F#127: Set-1 clitic must precede -ەوە on past transitive verbs
    assert any("Clitic position" in v for v in
               checker.check_sentence("کردەوەمان").violations)
    assert not any("Clitic position" in v for v in
                   checker.check_sentence("کردمانەوە").violations)
    assert not any("Clitic position" in v for v in
                   checker.check_sentence("ماڵەوەم خۆشە").violations)
    # F#122: ش/یش precedes the clitic on demonstratives; خۆ is exempt
    assert any("ش/یش order" in v for v in
               checker.check_sentence("ئەمەمیش باشە").violations)
    assert not any("ش/یش order" in v for v in
                   checker.check_sentence("خۆمیش هاتم").violations)
    print("  Category rules F#60/F#127/F#122 enforced")


def test_category_rules_f50_f155():
    """F#50 copular clitic agreement, F#155 dialectal participle."""
    checker = AgreementChecker()
    # F#50: verbless-sentence copula must match the subject pronoun
    assert any("Copular clitic mismatch" in v for v in
               checker.check_sentence("من کوردە").violations)
    assert any("Copular clitic mismatch" in v for v in
               checker.check_sentence("ئێمە کوردن").violations)
    for clean in ("من کوردم", "تۆ کوردی", "ئێمە کوردین", "ئەوان کوردن",
                  "من دەچم", "ئێمە ماڵمان هەیە"):
        assert not any("Copular clitic mismatch" in v for v in
                       checker.check_sentence(clean).violations), clean
    # F#155: dialectal ی/گ participle allomorphs on past stems
    assert any("Dialectal participle" in v for v in
               checker.check_sentence("ئەو هاتیە").violations)
    assert any("Dialectal participle" in v for v in
               checker.check_sentence("ئەو مردگە").violations)
    for clean in ("ئەو هاتووە", "ئەمە بەڵگە نییە", "ئەوە کوردیە"):
        assert not any("Dialectal participle" in v for v in
                       checker.check_sentence(clean).violations), clean
    print("  Category rules F#50/F#155 enforced")


def test_category_rules_f40_f123_f86():
    """F#40 perfect ە, F#123 demonstrative contraction, F#86 proper nouns."""
    checker = AgreementChecker()
    # R17/F#40: transitive perfect requires final ە
    assert any("Perfect missing copula" in v for v in
               checker.check_sentence("ئەو کتێبی گرتووم").violations)
    for clean in ("ئەو نامەکەی کردوویە", "من هاتووم", "ئێمە کەوتووین",
                  "خانووم جوانە"):
        assert not any("Perfect missing copula" in v for v in
                       checker.check_sentence(clean).violations), clean
    # F#123: بە/لە + demonstrative must contract
    assert any("Demonstrative contraction" in v for v in
               checker.check_sentence("بە ئەم پیاوە دەڵێم").violations)
    for clean in ("بەم پیاوە دەڵێم", "بۆ ئەو ماڵە چووم"):
        assert not any("Demonstrative contraction" in v for v in
                       checker.check_sentence(clean).violations), clean
    # F#86: proper nouns reject indefinite/plural markers
    assert any("Proper noun morphology" in v for v in
               checker.check_sentence("هەولێرێک بینیم").violations)
    assert any("Proper noun morphology" in v for v in
               checker.check_sentence("دهۆکان جوانن").violations)
    for clean in ("هەولێر جوانە", "هەولێرم خۆشدەوێت", "هەولێرییەکان هاتن"):
        assert not any("Proper noun morphology" in v for v in
                       checker.check_sentence(clean).violations), clean
    print("  Category rules F#40/F#123/F#86 enforced")


def test_mood_negation_rules_r14_f157_r15():
    """R14 نە/مە+ب ban, F#157 مە person restriction, R15 imperative ە."""
    checker = AgreementChecker()
    # R14: نە/مە replace the subjunctive/imperative ب
    assert any("Negation-ب" in v for v in
               checker.check_sentence("نەبچم بۆ ماڵەوە").violations)
    for clean in ("نەچم بۆ ماڵەوە", "ئەو نەبوو لێرە", "نەبم بە هاوڕێت",
                  "نەبینم", "نەبەم بۆ ماڵ"):
        assert not any("Negation-ب" in v for v in
                       checker.check_sentence(clean).violations), clean
    # F#157: prohibitive مە is 2nd-person only
    assert any("Prohibitive person" in v for v in
               checker.check_sentence("مەنووسم").violations)
    for clean in ("مەنووسە", "مەکە", "مەگرن", "مەزنم و باشم"):
        assert not any("Prohibitive person" in v for v in
                       checker.check_sentence(clean).violations), clean
    # R15/F#42: consonant-final imperative requires final ە
    assert any("Imperative missing" in v for v in
               checker.check_sentence("بنووس").violations)
    assert any("Imperative missing" in v for v in
               checker.check_sentence("مەگر").violations)
    for clean in ("بنووسە", "بگرە", "بخۆ", "بکە", "بگرن"):
        assert not any("Imperative missing" in v for v in
                       checker.check_sentence(clean).violations), clean
    print("  Mood/negation rules R14/F#157/R15 enforced")


def test_imperative_clitic_and_sh_zh_rules():
    """F#125 intransitive-imperative clitic ban, R19 ش→ژ present stems."""
    checker = AgreementChecker()
    # F#125: intransitive imperatives never host Set 1 clitics
    assert any("Imperative clitic" in v for v in
               checker.check_sentence("بمکەوە").violations)
    for clean in ("بمگرە", "بیخۆ", "بیخەوێنە", "بینووسە", "بتوانم"):
        assert not any("Imperative clitic" in v for v in
                       checker.check_sentence(clean).violations), clean
    # R19: present stem of کوشتن/هاوێشتن uses ژ
    assert any("Present stem" in v for v in
               checker.check_sentence("دەکوشم").violations)
    assert any("Present stem" in v for v in
               checker.check_sentence("دەهاوێشم").violations)
    for clean in ("دەکوژم", "دەیکوشت", "نەیکوشت", "کوشتی", "دەهاوێژم"):
        assert not any("Present stem" in v for v in
                       checker.check_sentence(clean).violations), clean
    print("  F#125 and R19 rules enforced")


def test_hebun_and_optative_rules():
    """F#72 possessed-noun هەبوون agreement, F#158 optative negation."""
    checker = AgreementChecker()
    # F#72: هەبوون agrees with the possessed noun
    assert any("Possessed-noun" in v for v in
               checker.check_sentence("کتێبەکانم هەیە").violations)
    assert any("Possessed-noun" in v for v in
               checker.check_sentence("کتێبەکەم هەن").violations)
    for clean in ("کتێبەکانم هەن", "کتێبەکەم هەیە", "ماڵمان هەیە",
                  "ئەو کارەکە نییە"):
        assert not any("Possessed-noun" in v for v in
                       checker.check_sentence(clean).violations), clean
    # F#158: optative negates with نە, never مە
    assert any("Optative negation" in v for v in
               checker.check_sentence("مەچووبام").violations)
    for clean in ("نەچووبام", "مەرحەبا", "مەکە"):
        assert not any("Optative negation" in v for v in
                       checker.check_sentence(clean).violations), clean
    print("  F#72 and F#158 rules enforced")


def test_negative_progressive_and_3sg_allomorph():
    """F#116 negative-progressive clitic shift, R12 3sg allomorphy."""
    checker = AgreementChecker()
    # F#116: agent clitic precedes دە under negation
    assert any("Negative progressive" in v for v in
               checker.check_sentence("نەدەمزانی").violations)
    for clean in ("نەمدەزانی", "نەیاندەکرد", "نەدەچووم"):
        assert not any("Negative progressive" in v for v in
                       checker.check_sentence(clean).violations), clean
    # R12: ە/ۆ-final stems take ات/وات in 3sg
    assert any("3sg allomorph" in v for v in
               checker.check_sentence("دەکەێت زۆر باش").violations)
    assert any("3sg allomorph" in v for v in
               checker.check_sentence("دەخۆێت").violations)
    for clean in ("دەکات زۆر باش", "دەخوات", "دەچێت بۆ ماڵ", "دەکەوێت"):
        assert not any("3sg allomorph" in v for v in
                       checker.check_sentence(clean).violations), clean
    print("  F#116 and R12 rules enforced")


def test_passive_and_clitic_stack_rules():
    """R13 present-stem passive, F#52 clitic stacking, F#124 ش ban."""
    checker = AgreementChecker()
    # R13: passive is built on the present stem
    assert any("Passive formation" in v for v in
               checker.check_sentence("نامەکە نووسترا").violations)
    assert any("Passive formation" in v for v in
               checker.check_sentence("پیاوەکە کوشترا").violations)
    for clean in ("نامەکە نووسرا", "پیاوەکە کوژرا", "دزەکە گیرا",
                  "کتێبەکە بەسترا", "خسترا"):
        assert not any("Passive formation" in v for v in
                       checker.check_sentence(clean).violations), clean
    # F#52: Set 2 clitics never stack
    assert any("Double Set 2" in v for v in
               checker.check_sentence("دەچمین بۆ شار").violations)
    for clean in ("دەچم بۆ شار", "دەچین بۆ شار", "یەکەمین جار",
                  "دووەمین ساڵ"):
        assert not any("Double Set 2" in v for v in
                       checker.check_sentence(clean).violations), clean
    # F#124 extension: هی rejects ش/یش too
    assert any("Clitic-barred" in v for v in
               checker.check_sentence("هیش باشە").violations)
    assert not any("Clitic-barred" in v for v in
                   checker.check_sentence("هی من باشە").violations)
    print("  R13, F#52 and F#124-ش rules enforced")


def test_nus_spelling_and_micro_rules():
    """F#164 نووس spelling, F#42 بچە exception, F#161 ئایا punctuation."""
    checker = AgreementChecker()
    # F#164: نووسین takes double وو
    assert any("و/وو spelling" in v for v in
               checker.check_sentence("دەنوسم").violations)
    assert any("و/وو spelling" in v for v in
               checker.check_sentence("نامەکە نوسراوە").violations)
    for clean in ("دەنووسم", "نامەکەم نووسی", "نوسخەیەکم هەیە"):
        assert not any("و/وو spelling" in v for v in
                       checker.check_sentence(clean).violations), clean
    # F#42 exception: imperative of چوون is بچۆ
    assert any("Imperative of چوون" in v for v in
               checker.check_sentence("بچە بۆ ماڵەوە").violations)
    for clean in ("بچۆ بۆ ماڵەوە", "بڕۆ بۆ ماڵەوە"):
        assert not any("Imperative of چوون" in v for v in
                       checker.check_sentence(clean).violations), clean
    # F#161: ئایا question closed with a period
    assert any("Interrogative punctuation" in v for v in
               checker.check_sentence("ئایا تۆ کوردیت.").violations)
    for clean in ("ئایا تۆ کوردیت؟", "ئایا تۆ کوردیت", "تۆ کوردیت."):
        assert not any("Interrogative punctuation" in v for v in
                       checker.check_sentence(clean).violations), clean
    print("  F#164, F#42-بچۆ and F#161 rules enforced")


def test_negated_clitic_position_and_dem_indef_frame():
    """F#39 negated transitive clitic order, R4 frame-final ێکە coverage."""
    checker = AgreementChecker()
    # F#39: under negation the agent clitic precedes the stem
    for bad in ("نەگرتم", "نەکردیان", "نامەکەم نەنووسیت"):
        assert any("Negated transitive clitic" in v for v in
                   checker.check_sentence(bad).violations), bad
    for clean in ("نەمگرت", "نەیانکرد", "نەهاتم", "نەچووم بۆ شار",
                  "نەگرتوومە", "نەگرتبام", "نەگرتن گرنگە", "نەمردم"):
        assert not any("Negated transitive clitic" in v for v in
                       checker.check_sentence(clean).violations), clean
    # R4 extension: frame-final *ئەم کتێبێکە now caught
    for bad in ("ئەم کتێبێکە", "ئەو کوڕێکە"):
        assert any("Demonstrative+indefinite" in v for v in
                   checker.check_sentence(bad).violations), bad
    for clean in ("ئەم کتێبە", "ئەو یەکێکە", "کتێبێکە لەسەر مێزەکە"):
        assert not any("Demonstrative+indefinite" in v for v in
                       checker.check_sentence(clean).violations), clean
    print("  F#39 and R4-frame rules enforced")


def test_past_agreement_and_np_frame_rules():
    """F#39/F#27 past subject agreement, F#87 order, F#77 numerals,
    F#10-frame, F#123-fused, F#90 double plural."""
    checker = AgreementChecker()
    # F#39/F#27: past intransitive Set 2 suffix matches the subject
    for bad in ("من هات", "ئەو ڕۆیشتی", "تۆ فڕی", "ئێمە فڕین"):
        assert any("Past subject-verb" in v for v in
                   checker.check_sentence(bad).violations), bad
    for clean in ("من هاتم", "ئەو هات", "تۆ هاتیت", "ئێمە هاتین",
                  "ئەوان هاتن", "بۆ من هات", "براکەی من هات"):
        assert not any("Past subject-verb" in v for v in
                       checker.check_sentence(clean).violations), clean
    # F#87 familiarity hierarchy in coordination
    assert any("Familiarity" in v for v in
               checker.check_sentence("ئازاد و من هاتین").violations)
    for clean in ("من و ئازاد هاتین", "من و تۆ دەچین",
                  "ئەو هات و من ڕۆیشتم"):
        assert not any("Familiarity" in v for v in
                       checker.check_sentence(clean).violations), clean
    # F#77 numeral subjects force plural
    assert any("Numeral subject" in v for v in
               checker.check_sentence("دوو کوڕ هات").violations)
    for clean in ("دوو کوڕ هاتن", "دوو ڕۆژ مایەوە", "دوو سێو دەخوات"):
        assert not any("Numeral subject" in v for v in
                       checker.check_sentence(clean).violations), clean
    # F#10-frame closing ە + F#123 fused time words
    for bad in ("لەم شار دەژیم", "ئەم کتێب باشە"):
        assert any("Demonstrative frame" in v for v in
                   checker.check_sentence(bad).violations), bad
    for clean in ("لەم شارە دەژیم", "ئەو نان دەخوات", "لەم ساتەدا وەرە"):
        assert not any("Demonstrative frame" in v for v in
                       checker.check_sentence(clean).violations), clean
    assert any("Fused time" in v for v in
               checker.check_sentence("ئەم ساڵ دەچم بۆ هەولێر").violations)
    assert not any("Fused time" in v for v in
                   checker.check_sentence("ئەم ساڵە باشە").violations)
    # F#90 double plural-definite
    assert any("Double plural-definite" in v for v in
               checker.check_sentence("کتێبەکانەکان هاتن").violations)
    assert not any("Double plural-definite" in v for v in
                   checker.check_sentence("کتێبەکان هاتن").violations)
    print("  Past agreement and NP-frame rules enforced")


def test_mood_negation_and_stem_micro_rules():
    """F#119, F#169, F#43, F#35, F#36, F#168, R18/F#46, F#76."""
    checker = AgreementChecker()
    # F#119: short wh-question closed with a period
    for bad in ("کێ هات.", "بۆچی وا دەکەیت."):
        assert any("F#119" in v for v in
                   checker.check_sentence(bad).violations), bad
    for clean in ("کێ هات؟", "کێ هات پێی بڵێ.", "نازانم کێ هات."):
        assert not any("F#119" in v for v in
                       checker.check_sentence(clean).violations), clean
    # F#169 unfused نە+ئە and F#43 stacked negation
    assert any("Unfused negation" in v for v in
               checker.check_sentence("نەئەچم بۆ شار").violations)
    assert not any("Unfused negation" in v for v in
                   checker.check_sentence("ناچم بۆ شار").violations)
    assert any("Double negation" in v for v in
               checker.check_sentence("نەنادەچم").violations)
    assert not any("Double negation" in v for v in
                   checker.check_sentence("نادەچم").violations)
    # F#35 happening verbs keep ێ
    for bad in ("دەسووتم", "دەشکین"):
        assert any("Happening-verb" in v for v in
                   checker.check_sentence(bad).violations), bad
    for clean in ("دەسووتێم", "دەشکێنم", "بشکێت"):
        assert not any("Happening-verb" in v for v in
                       checker.check_sentence(clean).violations), clean
    # F#36 suppletive causatives
    assert any("Causative formation" in v for v in
               checker.check_sentence("هاتاندی بۆ ماڵ").violations)
    for clean in ("هێنای بۆ ماڵ", "سووتاندی"):
        assert not any("Causative formation" in v for v in
                       checker.check_sentence(clean).violations), clean
    # F#168 خواردن keeps خۆ/خوا
    assert any("خواردن stem" in v for v in
               checker.check_sentence("دەخوێم").violations)
    for clean in ("دەخۆم", "دەخوێنم"):
        assert not any("خواردن stem" in v for v in
                       checker.check_sentence(clean).violations), clean
    # R18/F#46 preverb clitic position
    for bad in ("هەڵگرتمان", "داخستیان"):
        assert any("Preverb clitic" in v for v in
                   checker.check_sentence(bad).violations), bad
    for clean in ("هەڵمانگرت", "پێویستمان بە یارمەتییە",
                  "تێبینیمان کرد", "داگرتنمان"):
        assert not any("Preverb clitic" in v for v in
                       checker.check_sentence(clean).violations), clean
    # F#76 plural vocative demands plural imperative
    assert any("Vocative-imperative" in v for v in
               checker.check_sentence("کوڕینە بڕۆ").violations)
    for clean in ("کوڕینە بڕۆن", "کوڕینە بەرەو ماڵ بڕۆن"):
        assert not any("Vocative-imperative" in v for v in
                       checker.check_sentence(clean).violations), clean
    print("  Mood, negation and stem micro-rules enforced")


def test_fused_preposition_clitic_analysis():
    """F#26/R10: پێم/لێی/بۆمان analyze as ADP hosting a Set-1 clitic."""
    from src.morphology.analyzer import MorphologicalAnalyzer
    analyzer = MorphologicalAnalyzer(use_klpt=False)
    for tok, person in [("پێم", "1"), ("لێی", "3"), ("پێت", "2"), ("بۆمان", "1")]:
        feat = analyzer.analyze_token(tok)
        assert feat.pos == "ADP", f"{tok}: expected ADP, got {feat.pos}"
        assert feat.clitic_person == person, (
            f"{tok}: expected clitic person {person}, got {feat.clitic_person}"
        )
        assert feat.raw_analysis.get("fused_preposition"), tok
    # Non-clitic lookalikes stay untouched
    for tok in ("پێش", "لێو", "بۆن", "پێست"):
        feat = analyzer.analyze_token(tok)
        assert not feat.raw_analysis.get("fused_preposition"), (
            f"{tok} wrongly analyzed as fused preposition"
        )
    print("  Fused preposition+clitic analysis (F#26/R10) verified")


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

    print("\n=== Span-based Edit Tests (EVAL-1) ===")
    test_span_based_edits_substitution()
    test_span_based_edits_insertion()
    test_span_based_edits_deletion()
    test_span_based_edits_morphological_classification()
    test_span_based_edits_full_substitution()
    test_evaluate_corpus_span_basic()
    test_evaluate_corpus_span_empty()

    print("\n=== Sentence-level Metric Tests (EVAL-3) ===")
    test_evaluate_sentence_perfect()
    test_evaluate_sentence_no_correction()
    test_evaluate_corpus_with_sentences_basic()
    test_evaluate_corpus_with_sentences_mixed()
    test_evaluate_corpus_with_sentences_empty()

    print("\n=== Object-Verb Ergative & Fourteen-Check Tests ===")
    test_agreement_checker_object_verb_ergative()
    test_agreement_checker_fourteen_checks_counted()

    print("\n=== PIPE-4: Per-Law Agreement & CER Tests ===")
    test_evaluate_agreement_by_check_basic()
    test_evaluate_agreement_by_check_per_law_keys()
    test_compute_cer_identical()
    test_compute_cer_different()

    print("\nAll evaluation tests passed!")


# ============================================================================
# PIPE-4: Per-Agreement-Law Breakdown Tests
# ============================================================================

def test_evaluate_agreement_by_check_basic():
    """Per-check breakdown runs and returns dict with expected structure."""
    sentences = ["من دەچم بۆ قوتابخانە", "ئەوان دەچن بۆ بازاڕ"]
    result = evaluate_agreement_by_check(sentences)
    assert "per_check" in result
    assert "per_law" in result
    assert len(result["per_check"]) == 15  # 15 agreement checks
    for label, info in result["per_check"].items():
        assert "accuracy" in info
        assert "total" in info
        # Denominator counts only sentences the check applies to, so it
        # ranges 0..2 — inapplicable checks are not silent passes.
        assert 0 <= info["total"] <= 2
    # Both sentences are pronoun+present-verb, so subject-verb applies twice.
    assert result["per_check"]["subject_verb"]["total"] == 2
    print(f"  Per-check breakdown: {len(result['per_check'])} checks")


def test_evaluate_agreement_by_check_per_law_keys():
    """Per-law summary contains Law 1 and Law 2."""
    sentences = ["من دەچم بۆ قوتابخانە"]
    result = evaluate_agreement_by_check(sentences)
    per_law = result["per_law"]
    assert "Law 1" in per_law, f"Expected 'Law 1' in per_law, got keys: {list(per_law.keys())}"
    assert "Law 2" in per_law, f"Expected 'Law 2' in per_law, got keys: {list(per_law.keys())}"
    for law, info in per_law.items():
        assert 0.0 <= info["accuracy"] <= 1.0
    print(f"  Per-law: Law 1={per_law['Law 1']['accuracy']:.2f}, Law 2={per_law['Law 2']['accuracy']:.2f}")


# ============================================================================
# PIPE-4: CER Tests
# ============================================================================

def _compute_cer(hypotheses, references):
    """Local copy of CER algorithm for testing (mirrors 07_evaluate.compute_cer)."""
    total_dist, total_len = 0, 0
    for hyp, ref in zip(hypotheses, references):
        m, n = len(hyp), len(ref)
        dp = list(range(n + 1))
        for i in range(1, m + 1):
            prev = dp[0]
            dp[0] = i
            for j in range(1, n + 1):
                temp = dp[j]
                if hyp[i - 1] == ref[j - 1]:
                    dp[j] = prev
                else:
                    dp[j] = 1 + min(prev, dp[j], dp[j - 1])
                prev = temp
        total_dist += dp[n]
        total_len += max(n, 1)
    return total_dist / total_len if total_len > 0 else 0.0


def test_compute_cer_identical():
    """CER of identical strings should be 0."""
    assert _compute_cer(["abc"], ["abc"]) == 0.0
    print("  CER identical: 0.0")


def test_compute_cer_different():
    """CER of completely different strings should be > 0."""
    cer = _compute_cer(["abc"], ["xyz"])
    assert cer == 1.0  # all 3 chars wrong out of 3 ref chars
    print(f"  CER different: {cer}")
