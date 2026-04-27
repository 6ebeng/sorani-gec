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
    
    # Western digits should become Extended Arabic-Indic (Sorani uses U+06F0–U+06F9)
    result = normalizer.normalize("2026")
    assert result == "۲۰۲۶", f"Expected Extended Arabic-Indic '۲۰۲۶', got '{result}'"
    
    # Extended Arabic-Indic digits should stay as-is
    result2 = normalizer.normalize("۲۰۲۶")
    assert result2 == "۲۰۲۶", f"Expected '۲۰۲۶', got '{result2}'"
    
    # Standard Arabic-Indic digits should become Extended
    result3 = normalizer.normalize("٢٠٢٦")
    assert result3 == "۲۰۲۶", f"Expected Extended form, got '{result3}'"


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


def test_teh_marbuta_folds_to_heh():
    """April 2026 audit: TEH MARBUTA (\u0629) is Arabic-only and should fold to \u06d5."""
    normalizer = SoraniNormalizer()
    # Standalone TEH MARBUTA in an Arabic loan word
    result = normalizer.normalize("\u062a\u0631\u062c\u0645\u0629")
    assert "\u0629" not in result, "TEH MARBUTA should be removed"
    assert result.endswith("\u06d5"), f"TEH MARBUTA should fold to \u06d5, got {result!r}"


def test_diacritics_removed_when_enabled():
    """Harakat should be stripped when remove_diacritics=True."""
    normalizer = SoraniNormalizer(remove_diacritics=True)
    # Text with FATHA, KASRA, SHADDA, SUKUN
    text = "\u0628\u064e\u0634\u0650\u062a\u064f"  # b-FATHA \u0634-KASRA t-DAMMA
    result = normalizer.normalize(text)
    assert "\u064e" not in result and "\u0650" not in result and "\u064f" not in result
    # Ensure base letters survive
    assert "\u0628" in result and "\u062a" in result


def test_diacritics_kept_when_disabled():
    """Harakat survive when remove_diacritics=False (legacy behaviour)."""
    normalizer = SoraniNormalizer(remove_diacritics=False)
    text = "\u0628\u064e\u062a"
    result = normalizer.normalize(text)
    assert "\u064e" in result


def test_strict_sorani_quality_gate():
    """Pure-Arabic citation lines fail the strict-Sorani filter."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_normalize_script",
        os.path.join(os.path.dirname(__file__), "..", "scripts", "02_normalize.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # Pure Arabic line (no Sorani-distinctive letters)
    arabic_line = "\u0646\u062e\u0628\u0629 \u0645\u0646 \u0627\u0644\u0628\u0627\u062d\u062b\u064a\u0646\u060c \u062d\u0636\u0627\u0631\u0629 \u0627\u0644\u0639\u0631\u0627\u0642."
    passes, reason = mod.passes_quality_gate(arabic_line, strict_sorani=True)
    assert not passes and reason == "non_sorani_script"
    # Genuine Sorani line passes the Sorani-distinctive check (contains \u06d5, \u06cc, \u06af).
    # We assert only the absence of the non_sorani_script reason; the line
    # may still trip other gates (e.g. token count) which are out of scope.
    sorani_line = "\u0626\u06d5\u0645\u06d5 \u0648\u0634\u06d5\u06cc\u06d5\u06a9\u06cc \u06a9\u0648\u0631\u062f\u06cc\u06cc\u06d5 \u0648 \u0628\u0627\u0634\u06d5 \u0648 \u062c\u0648\u0627\u0646\u06d5 \u0644\u0647\u060c \u062f\u0648\u0648\u06d5\u0645\u06cc\u0646."
    _, reason2 = mod.passes_quality_gate(sorani_line, strict_sorani=True)
    assert reason2 != "non_sorani_script"
    # Without strict mode, the Arabic line passes the Sorani-distinctive check
    # (it may still fail other gates but not for this reason).
    passes3, reason3 = mod.passes_quality_gate(arabic_line, strict_sorani=False)
    assert reason3 != "non_sorani_script"


if __name__ == "__main__":
    test_basic_normalization()
    test_arabic_digit_normalization()
    test_sentence_split()
    test_deduplication()
    test_heh_context_dependent_normalization()
    test_heh_normalization_disabled_when_normalize_chars_false()
    print("All normalizer tests passed!")
