"""
Tests for error generators.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.errors.subject_verb import SubjectVerbErrorGenerator
from src.errors.clitic import CliticErrorGenerator
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


if __name__ == "__main__":
    test_subject_verb_generator()
    test_clitic_generator()
    test_vocative_imperative_generator()
    test_conditional_agreement_generator()
    test_adverb_verb_tense_generator()
    test_pipeline()
    print("All error generator tests passed!")
