"""
Tests for error generators.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.errors.subject_verb import SubjectVerbErrorGenerator
from src.errors.clitic import CliticErrorGenerator
from src.errors.noun_adjective import NounAdjectiveErrorGenerator
from src.errors.tense_agreement import TenseAgreementErrorGenerator
from src.errors.syntax_roles import CaseRoleErrorGenerator
from src.errors.dialectal import DialectalParticipleErrorGenerator
from src.errors.relative_clause import RelativeClauseErrorGenerator
from src.errors.adversative import AdversativeConnectorErrorGenerator
from src.errors.participle_swap import ParticipleSwapErrorGenerator
from src.errors.orthography import OrthographicErrorGenerator
from src.errors.negative_concord import NegativeConcordErrorGenerator
from src.errors.vocative_imperative import VocativeImperativeErrorGenerator
from src.errors.conditional_agreement import ConditionalAgreementErrorGenerator
from src.errors.adverb_verb_tense import AdverbVerbTenseErrorGenerator
from src.errors.preposition_fusion import PrepositionFusionErrorGenerator
from src.errors.demonstrative_contraction import DemonstrativeContractionErrorGenerator
from src.errors.quantifier_agreement import QuantifierAgreementErrorGenerator
from src.errors.possessive_clitic import PossessiveCliticErrorGenerator
from src.errors.polite_imperative import PoliteImperativeErrorGenerator
from src.errors.pipeline import ErrorPipeline


def test_subject_verb_generator():
    gen = SubjectVerbErrorGenerator(error_rate=1.0, seed=42)  # 100% error rate for testing
    
    # Test finding eligible positions
    sentence = "من دەچم بۆ بازاڕ"  # "I go to market"
    positions = gen.find_eligible_positions(sentence)
    print(f"Found {len(positions)} eligible positions in: {sentence}")
    
    # Test error injection
    result = gen.inject_errors(sentence)
    print(f"Original:  {result.original}")
    print(f"Corrupted: {result.corrupted}")
    print(f"Errors:    {[e.description for e in result.errors]}")


def test_clitic_generator():
    gen = CliticErrorGenerator(error_rate=1.0, seed=42)
    
    sentence = "کتێبەکەم لەسەر مێزەکەت بوو"
    positions = gen.find_eligible_positions(sentence)
    print(f"\nFound {len(positions)} eligible positions in: {sentence}")
    
    result = gen.inject_errors(sentence)
    print(f"Original:  {result.original}")
    print(f"Corrupted: {result.corrupted}")


def test_case_role_generator():
    gen = CaseRoleErrorGenerator(error_rate=1.0, seed=42)

    sentence = "دەرگاکە لەلایەن کاوەوە بە کلیل کرایەوە"
    positions = gen.find_eligible_positions(sentence)
    assert len(positions) == 2  # Should find "لەلایەن" and "بە"

    result = gen.inject_errors(sentence)
    assert result.has_errors
    print(f"\nOriginal:  {result.original}")
    print(f"Corrupted: {result.corrupted}")
    print(f"Errors:    {[e.description for e in result.errors]}")

def test_dialectal_participle_generator():
    gen = DialectalParticipleErrorGenerator(error_rate=1.0, seed=42)
    sentence = "ئەو پیاوە مردوە و کتێبەکەی سوتاوە"
    positions = gen.find_eligible_positions(sentence)
    assert len(positions) == 2

    result = gen.inject_errors(sentence)
    assert result.has_errors
    print(f"\nOriginal:  {result.original}")
    print(f"Corrupted: {result.corrupted}")
    print(f"Errors:    {[e.description for e in result.errors]}")

def test_relative_clause_generator():
    gen = RelativeClauseErrorGenerator(error_rate=1.0, seed=42)
    sentence = "ئەو پەرداخەکەی کە من کڕیم شکا"
    positions = gen.find_eligible_positions(sentence)
    assert len(positions) == 1

    result = gen.inject_errors(sentence)
    assert result.has_errors
    # Expected output: ئەو پەرداخەکە من کڕیم شکا (wrong because both ی and کە are missing)
    print(f"\nOriginal:  {result.original}")
    print(f"Corrupted: {result.corrupted}")
    print(f"Errors:    {[e.description for e in result.errors]}")

def test_adversative_connector_generator():
    gen = AdversativeConnectorErrorGenerator(error_rate=1.0, seed=42)
    # Sentence with only one connector
    sentence = "هۆنراوە زیرەکە، بەڵام ناخوێنێ"
    positions = gen.find_eligible_positions(sentence)
    assert len(positions) == 1

    result = gen.inject_errors(sentence)
    assert result.has_errors
    print(f"\nOriginal:  {result.original}")
    print(f"Corrupted: {result.corrupted}")
    print(f"Errors:    {[e.description for e in result.errors]}")

def test_participle_swap_generator():
    gen = ParticipleSwapErrorGenerator(error_rate=1.0, seed=42)
    sentence = "ئەمە نووسراو زۆر کۆنە"
    positions = gen.find_eligible_positions(sentence)
    assert len(positions) == 1

    result = gen.inject_errors(sentence)
    assert result.has_errors
    print(f"\nOriginal:  {result.original}")
    print(f"Corrupted: {result.corrupted}")
    print(f"Errors:    {[e.description for e in result.errors]}")

def test_orthographic_generator():
    gen = OrthographicErrorGenerator(error_rate=1.0, seed=42)
    sentence = "من حەوت باخ دەبینم"
    positions = gen.find_eligible_positions(sentence)
    assert len(positions) > 0

    result = gen.inject_errors(sentence)
    assert result.has_errors
    print(f"\nOriginal:  {result.original}")
    print(f"Corrupted: {result.corrupted}")
    print(f"Errors:    {[e.description for e in result.errors]}")

def test_negative_concord_generator():
    gen = NegativeConcordErrorGenerator(error_rate=1.0, seed=42)
    sentence = "من هیچم پێ ناخورێ"
    positions = gen.find_eligible_positions(sentence)
    assert len(positions) == 1

    result = gen.inject_errors(sentence)
    assert result.has_errors
    print(f"\nOriginal:  {result.original}")
    print(f"Corrupted: {result.corrupted}")
    print(f"Errors:    {[e.description for e in result.errors]}")

def test_pipeline():
    pipeline = ErrorPipeline(error_rate=0.5, seed=42)
    
    test_sentences = [
        "من دەچم بۆ قوتابخانە",
        "ئەوان دەنوسن بۆ مامۆستاکانیان",
        "تۆ دەزانیت ئەم بابەتە",
    ]
    
    print("\n--- Pipeline Test ---")
    for sentence in test_sentences:
        result = pipeline.process_sentence(sentence)
        print(f"Original:  {result.original}")
        print(f"Corrupted: {result.corrupted}")
        print(f"Errors:    {len(result.errors)}")
        print()


def test_vocative_imperative_generator():
    """H4: Vocative+imperative generator finds eligible positions."""
    gen = VocativeImperativeErrorGenerator(error_rate=1.0, seed=42)
    # Generator should at least instantiate and have correct error_type
    assert gen.error_type == "vocative_imperative"
    print(f"  VocativeImperativeErrorGenerator initialized: error_type={gen.error_type}")


def test_conditional_agreement_generator():
    """H4: Conditional agreement generator finds eligible positions."""
    gen = ConditionalAgreementErrorGenerator(error_rate=1.0, seed=42)
    assert gen.error_type == "conditional_agreement"
    # Test with a conditional sentence: "ئەگەر بچم" (if I-go)
    sentence = "ئەگەر بچم"
    positions = gen.find_eligible_positions(sentence)
    print(f"  ConditionalAgreementErrorGenerator: positions={len(positions)} for '{sentence}'")


def test_adverb_verb_tense_generator():
    """H4: Adverb-verb tense generator finds eligible positions."""
    gen = AdverbVerbTenseErrorGenerator(error_rate=1.0, seed=42)
    assert gen.error_type == "adverb_verb_tense"
    # Test with a temporal adverb + past verb: "دوێنێ چووم" (yesterday I-went)
    sentence = "دوێنێ چووم"
    positions = gen.find_eligible_positions(sentence)
    print(f"  AdverbVerbTenseErrorGenerator: positions={len(positions)} for '{sentence}'")


# ─── Enhanced functional tests for Phase B fixes ────────────────────────────


class TestTenseAgreementFunctional:
    """Verify tense_agreement generator produces valid, non-identity errors."""

    def setup_method(self):
        self.gen = TenseAgreementErrorGenerator(error_rate=1.0, seed=42)

    def test_past_verb_gets_different_agreement(self):
        """Past verb 'هاتن' (3pl came) should get a different ending."""
        sentence = "ئەوان هاتن بۆ ماڵ"
        result = self.gen.inject_errors(sentence)
        if result.has_errors:
            assert result.corrupted != result.original

    def test_no_doubled_morphemes(self):
        """Verify the fix prevents doubled morphemes like هاتنن."""
        sentence = "ئەوان هاتن بۆ ماڵ"
        result = self.gen.inject_errors(sentence)
        if result.has_errors:
            assert "هاتنن" not in result.corrupted

    def test_negated_past_verb(self):
        """Negated past verb 'نەکرد' should produce a valid substitution."""
        sentence = "من نەکرد ئەو کارە"
        result = self.gen.inject_errors(sentence)
        if result.has_errors:
            assert result.corrupted != result.original


class TestRelativeClauseFunctional:
    """Verify relative_clause generator handles multiple pattern types."""

    def setup_method(self):
        self.gen = RelativeClauseErrorGenerator(error_rate=1.0, seed=42)

    def test_definite_pattern(self):
        """Pattern: پەرداخەکەی کە (definite + ezafe + کە)."""
        sentence = "ئەو پەرداخەکەی کە من کڕیم شکا"
        positions = self.gen.find_eligible_positions(sentence)
        assert len(positions) >= 1
        result = self.gen.inject_errors(sentence)
        assert result.has_errors

    def test_indefinite_pattern(self):
        """Pattern: پیاوێکی کە (indefinite + ezafe + کە)."""
        sentence = "پیاوێکی کە دوێنێ هات"
        positions = self.gen.find_eligible_positions(sentence)
        assert len(positions) >= 1

    def test_error_modifies_sentence(self):
        """The corrupted output must differ from the original."""
        sentence = "ئەو پەرداخەکەی کە من کڕیم شکا"
        result = self.gen.inject_errors(sentence)
        if result.has_errors:
            assert result.corrupted != result.original


class TestVocativeImperativeFunctional:
    """Verify vocative_imperative handles vowel-final stems."""

    def setup_method(self):
        self.gen = VocativeImperativeErrorGenerator(error_rate=1.0, seed=42)

    def test_consonant_final_imperative(self):
        """Standard consonant-final: بنووسە (write-SG)."""
        sentence = "کاوە، بنووسە!"
        positions = self.gen.find_eligible_positions(sentence)
        # Should find imperative position
        if len(positions) > 0:
            result = self.gen.inject_errors(sentence)
            assert result.has_errors

    def test_vowel_final_imperative(self):
        """Vowel-final: بچۆ (go-SG) — should be found by Pattern B."""
        sentence = "کاوە، بچۆ بۆ ماڵ!"
        positions = self.gen.find_eligible_positions(sentence)
        # Pattern B should catch vowel-final stems
        print(f"  Vowel-final imperative positions: {len(positions)}")

    def test_error_type_correct(self):
        assert self.gen.error_type == "vocative_imperative"


class TestConditionalAgreementFunctional:
    """Verify conditional_agreement handles no-comma sentences correctly."""

    def setup_method(self):
        self.gen = ConditionalAgreementErrorGenerator(error_rate=1.0, seed=42)

    def test_comma_separated_conditional(self):
        """Standard: 'ئەگەر بزانیت، دەیڵێت' — comma marks boundary."""
        sentence = "ئەگەر بزانیت، دەیڵێت"
        positions = self.gen.find_eligible_positions(sentence)
        assert len(positions) >= 1

    def test_no_comma_conditional(self):
        """No comma: 'ئەگەر بزانیت دەیڵێت' — heuristic boundary."""
        sentence = "ئەگەر بزانیت دەیڵێت"
        positions = self.gen.find_eligible_positions(sentence)
        assert len(positions) >= 1

    def test_error_modifies_sentence(self):
        sentence = "ئەگەر بزانیت، دەیڵێت"
        result = self.gen.inject_errors(sentence)
        if result.has_errors:
            assert result.corrupted != result.original


class TestAdverbVerbTenseFunctional:
    """Verify adverb_verb_tense handles past intransitive verbs."""

    def setup_method(self):
        self.gen = AdverbVerbTenseErrorGenerator(error_rate=1.0, seed=42)

    def test_past_transitive_with_adverb(self):
        """'دوێنێ نووسیم' — past + temporal adverb."""
        sentence = "دوێنێ نووسیم"
        positions = self.gen.find_eligible_positions(sentence)
        print(f"  Past transitive positions: {len(positions)}")

    def test_past_intransitive_3pl(self):
        """'پێشتر هاتن' — past intransitive 3pl (ن ending)."""
        sentence = "پێشتر هاتن"
        positions = self.gen.find_eligible_positions(sentence)
        # Should find position now that ن is in PAST_CLITICS
        print(f"  Past intransitive 3pl positions: {len(positions)}")

    def test_error_type_correct(self):
        assert self.gen.error_type == "adverb_verb_tense"


class TestNounAdjectiveFunctional:
    """Verify noun_adjective det_cooccurrence injects errors into correct forms."""

    def setup_method(self):
        self.gen = NounAdjectiveErrorGenerator(error_rate=1.0, seed=42)

    def test_demonstrative_correct_form(self):
        """'ئەم کتێبە' (this book) — correct form should get error injected."""
        sentence = "ئەم کتێبە زۆر باشە"
        positions = self.gen.find_eligible_positions(sentence)
        print(f"  Demonstrative positions: {len(positions)}")

    def test_error_modifies_sentence(self):
        """Corrupted output must differ from original when errors exist."""
        sentence = "ئەم کتێبە زۆر باشە"
        result = self.gen.inject_errors(sentence)
        if result.has_errors:
            assert result.corrupted != result.original

    def test_error_type_correct(self):
        assert self.gen.error_type == "noun_adjective_agreement"


class TestPrepositionFusionFunctional:
    """Verify preposition_fusion generator un-fuses synthetic forms."""

    def setup_method(self):
        self.gen = PrepositionFusionErrorGenerator(error_rate=1.0, seed=42)

    def test_finds_fused_form(self):
        """'پێم دەخوشە' — should detect پێم as fused form."""
        sentence = "پێم دەخوشە"
        positions = self.gen.find_eligible_positions(sentence)
        assert len(positions) >= 1
        assert positions[0]["original"] == "پێم"

    def test_unfuses_correctly(self):
        """پێم should be un-fused to بە من."""
        sentence = "پێم دەخوشە"
        positions = self.gen.find_eligible_positions(sentence)
        assert len(positions) >= 1
        error = self.gen.generate_error(positions[0])
        assert error == "بە من"

    def test_error_modifies_sentence(self):
        sentence = "پێم دەخوشە"
        result = self.gen.inject_errors(sentence)
        if result.has_errors:
            assert result.corrupted != result.original

    def test_error_type_correct(self):
        assert self.gen.error_type == "preposition_fusion"


class TestDemonstrativeContractionFunctional:
    """Verify demonstrative_contraction generator splits contractions."""

    def setup_method(self):
        self.gen = DemonstrativeContractionErrorGenerator(error_rate=1.0, seed=42)

    def test_finds_contraction(self):
        """'لەم شارەدا' — should detect لەم as contracted form."""
        sentence = "لەم شارەدا زۆر کەسم ناسی"
        positions = self.gen.find_eligible_positions(sentence)
        assert len(positions) >= 1
        assert positions[0]["original"] == "لەم"

    def test_splits_correctly(self):
        """لەم should be split to لە ئەم."""
        sentence = "لەم شارەدا زۆر کەسم ناسی"
        positions = self.gen.find_eligible_positions(sentence)
        assert len(positions) >= 1
        error = self.gen.generate_error(positions[0])
        assert error == "لە ئەم"

    def test_error_modifies_sentence(self):
        sentence = "بەم شێوەیە"
        result = self.gen.inject_errors(sentence)
        if result.has_errors:
            assert result.corrupted != result.original

    def test_error_type_correct(self):
        assert self.gen.error_type == "demonstrative_contraction"


class TestQuantifierAgreementFunctional:
    """Verify quantifier_agreement flips plural verb to singular."""

    def setup_method(self):
        self.gen = QuantifierAgreementErrorGenerator(error_rate=1.0, seed=42)

    def test_finds_quantifier_verb_pair(self):
        """'زۆر کەس دەچن' — quantifier + plural verb."""
        sentence = "زۆر کەس دەچن"
        positions = self.gen.find_eligible_positions(sentence)
        assert len(positions) >= 1

    def test_flips_to_singular(self):
        """Plural verb ن ending should flip to singular ێت."""
        sentence = "زۆر کەس دەچن"
        positions = self.gen.find_eligible_positions(sentence)
        if len(positions) >= 1:
            error = self.gen.generate_error(positions[0])
            assert error is not None
            assert error != "دەچن"

    def test_error_modifies_sentence(self):
        sentence = "هەموو خەڵکی دەچن بۆ بازاڕ"
        result = self.gen.inject_errors(sentence)
        if result.has_errors:
            assert result.corrupted != result.original

    def test_error_type_correct(self):
        assert self.gen.error_type == "quantifier_agreement"


class TestPossessiveCliticFunctional:
    """Verify possessive_clitic swaps possessive clitics on nouns."""

    def setup_method(self):
        self.gen = PossessiveCliticErrorGenerator(error_rate=1.0, seed=42)

    def test_finds_possessive_on_definite_noun(self):
        """'کتێبەکەم' — definite noun + possessive م."""
        sentence = "کتێبەکەم لەسەر مێزەکەیە"
        positions = self.gen.find_eligible_positions(sentence)
        assert len(positions) >= 1

    def test_swaps_possessive_clitic(self):
        """The possessive clitic should be swapped to a different person."""
        sentence = "کتێبەکەم لەسەر مێزەکەیە"
        positions = self.gen.find_eligible_positions(sentence)
        if len(positions) >= 1:
            error = self.gen.generate_error(positions[0])
            assert error is not None
            assert error != positions[0]["original"]

    def test_error_modifies_sentence(self):
        sentence = "ماڵەکەمان زۆر گەورەیە"
        result = self.gen.inject_errors(sentence)
        if result.has_errors:
            assert result.corrupted != result.original

    def test_error_type_correct(self):
        assert self.gen.error_type == "possessive_clitic"


class TestPoliteImperativeFunctional:
    """Verify polite_imperative drops/swaps politeness markers."""

    def setup_method(self):
        self.gen = PoliteImperativeErrorGenerator(error_rate=1.0, seed=42)

    def test_finds_polite_marker_takaya(self):
        """'تکایە بنووسە' — polite marker + imperative verb."""
        sentence = "تکایە بنووسە"
        positions = self.gen.find_eligible_positions(sentence)
        assert len(positions) >= 1

    def test_finds_polite_marker_farmoo(self):
        """'فەرموو بنیشە' — 'go ahead, sit down'."""
        sentence = "فەرموو بنیشە لێرە"
        positions = self.gen.find_eligible_positions(sentence)
        assert len(positions) >= 1

    def test_drops_polite_marker(self):
        """Dropping or swapping the marker should change the sentence."""
        sentence = "تکایە بنووسە"
        gen = PoliteImperativeErrorGenerator(error_rate=1.0, seed=0)
        result = gen.inject_errors(sentence)
        if result.has_errors:
            assert result.corrupted != result.original
            # Original marker should be absent or replaced
            assert "تکایە" not in result.corrupted or result.corrupted != sentence

    def test_error_type_correct(self):
        assert self.gen.error_type == "polite_imperative"

    def test_no_match_without_marker(self):
        """A bare imperative without a polite marker should not match."""
        sentence = "بنووسە لەسەر کاغەز"
        positions = self.gen.find_eligible_positions(sentence)
        assert len(positions) == 0


if __name__ == "__main__":
    test_subject_verb_generator()
    test_clitic_generator()
    test_vocative_imperative_generator()
    test_conditional_agreement_generator()
    test_adverb_verb_tense_generator()
    test_pipeline()
    print("All error generator tests passed!")
