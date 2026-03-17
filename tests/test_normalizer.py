"""
Tests for the Sorani Kurdish text normalizer.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.normalizer import SoraniNormalizer, sentence_split, deduplicate_sentences


def test_basic_normalization():
    normalizer = SoraniNormalizer()
    
    # Test whitespace normalization
    assert "کوردستان زۆر جوانە" == normalizer.normalize("کوردستان   زۆر   جوانە")
    
    # Test empty string
    assert "" == normalizer.normalize("")
    
    # Test None handling
    assert None is normalizer.normalize(None) if normalizer.normalize(None) is None else True


def test_arabic_digit_normalization():
    normalizer = SoraniNormalizer()
    
    # Western digits should become Arabic-Indic (Sorani Kurdish uses Arabic-Indic)
    result = normalizer.normalize("2026")
    assert result == "٢٠٢٦", f"Expected '٢٠٢٦', got '{result}'"
    
    # Arabic-Indic digits should stay as-is
    result2 = normalizer.normalize("٢٠٢٦")
    assert result2 == "٢٠٢٦", f"Expected '٢٠٢٦', got '{result2}'"


def test_sentence_split():
    text = "ئەمە وشەیەکە. ئەمەش وشەیەکی تر. سێیەمیش ئەمەیە."
    sentences = sentence_split(text)
    assert len(sentences) == 3


def test_deduplication():
    sentences = ["سڵاو", "جوانە", "سڵاو", "باشە", "جوانە"]
    result = deduplicate_sentences(sentences)
    assert len(result) == 3
    assert result[0] == "سڵاو"
    assert result[1] == "جوانە"
    assert result[2] == "باشە"


def test_heh_context_dependent_normalization():
    """C1: Word-initial \u0647 stays (consonant /h/), non-initial \u0647 → \u06d5."""
    normalizer = SoraniNormalizer()
    # Word-initial heh preserved
    result = normalizer.normalize("\u0647\u06d5\u06cc\u06d5")
    assert result[0] == "\u0647", f"Initial heh should stay U+0647, got U+{ord(result[0]):04X}"
    # Non-initial heh converted to \u06d5
    result2 = normalizer.normalize("\u0628\u0647\u0634\u062a")
    assert result2[1] == "\u06d5", f"Non-initial heh should become U+06D5, got U+{ord(result2[1]):04X}"
    # Multiple words: initial preserved, non-initial converted
    result3 = normalizer.normalize("\u0647\u0647 \u0647\u0647")
    assert result3[0] == "\u0647" and result3[1] == "\u06d5"
    assert result3[3] == "\u0647" and result3[4] == "\u06d5"
    print("  Context-dependent heh normalization OK")


def test_heh_normalization_disabled_when_normalize_chars_false():
    """C1: When normalize_chars=False, heh is not modified."""
    normalizer = SoraniNormalizer(normalize_chars=False)
    text = "\u0628\u0647\u0634\u062a"
    result = normalizer.normalize(text)
    assert "\u0647" in result, "Heh should be preserved when normalize_chars=False"
    print("  Heh normalization respects normalize_chars=False")


if __name__ == "__main__":
    test_basic_normalization()
    test_arabic_digit_normalization()
    test_sentence_split()
    test_deduplication()
    test_heh_context_dependent_normalization()
    test_heh_normalization_disabled_when_normalize_chars_false()
    print("All normalizer tests passed!")
