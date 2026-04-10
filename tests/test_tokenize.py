"""
Tests for the Sorani Kurdish tokenization utilities.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.tokenize import sorani_tokenize, sorani_word_tokenize, sorani_join


# ---------------------------------------------------------------------------
# sorani_tokenize — regex-based
# ---------------------------------------------------------------------------

def test_tokenize_basic_sorani():
    tokens = sorani_tokenize("کوردستان زۆر جوانە")
    assert tokens == ["کوردستان", "زۆر", "جوانە"]


def test_tokenize_strips_zwj():
    """ZWJ (U+200D) must be stripped before tokenization."""
    text = "کورد\u200dستان"
    tokens = sorani_tokenize(text)
    assert tokens == ["کوردستان"]


def test_tokenize_preserves_zwnj():
    """ZWNJ (U+200C) marks morpheme boundaries and must survive."""
    text = "دە\u200cنوسم"
    tokens = sorani_tokenize(text)
    assert len(tokens) == 1
    assert "\u200c" in tokens[0]


def test_tokenize_splits_conjunctive_waw():
    """Conjunctive و glued to next word should become two tokens."""
    tokens = sorani_tokenize("وئەو")
    assert tokens == ["و", "ئەو"], f"Got {tokens}"


def test_tokenize_standalone_waw_untouched():
    """Standalone و with space after it is not split further."""
    tokens = sorani_tokenize("من و تۆ")
    assert tokens == ["من", "و", "تۆ"]


def test_tokenize_punctuation_separated():
    tokens = sorani_tokenize("سڵاو، چۆنی؟")
    assert "،" in tokens
    assert "؟" in tokens
    assert tokens[0] == "سڵاو"


def test_tokenize_mixed_scripts():
    tokens = sorani_tokenize("Python زمانێکی ۲۰۲۶")
    assert "Python" in tokens
    assert "۲۰۲۶" in tokens
    assert "زمانێکی" in tokens


def test_tokenize_empty_string():
    assert sorani_tokenize("") == []


def test_tokenize_whitespace_only():
    assert sorani_tokenize("   \t\n  ") == []


def test_tokenize_extended_arabic_indic_digits():
    tokens = sorani_tokenize("ژمارە ۱۲۳")
    assert "۱۲۳" in tokens


def test_tokenize_western_digits():
    tokens = sorani_tokenize("ژمارە 456")
    assert "456" in tokens


# ---------------------------------------------------------------------------
# sorani_word_tokenize — whitespace-based
# ---------------------------------------------------------------------------

def test_word_tokenize_basic():
    tokens = sorani_word_tokenize("کوردستان زۆر جوانە")
    assert tokens == ["کوردستان", "زۆر", "جوانە"]


def test_word_tokenize_strips_zwj():
    text = "کورد\u200dستان زۆر"
    tokens = sorani_word_tokenize(text)
    assert tokens[0] == "کوردستان"


def test_word_tokenize_empty():
    assert sorani_word_tokenize("") == []


def test_word_tokenize_keeps_punctuation_attached():
    """Unlike sorani_tokenize, word tokenizer keeps punctuation on words."""
    tokens = sorani_word_tokenize("سڵاو، چۆنی؟")
    assert tokens == ["سڵاو،", "چۆنی؟"]


# ---------------------------------------------------------------------------
# sorani_join — token rejoining
# ---------------------------------------------------------------------------

def test_join_basic():
    result = sorani_join(["سڵاو", "جیهان"])
    assert result == "سڵاو جیهان"


def test_join_suppresses_space_before_punctuation():
    result = sorani_join(["سڵاو", "،", "جیهان"])
    assert result == "سڵاو، جیهان"


def test_join_question_mark():
    result = sorani_join(["چۆنی", "؟"])
    assert result == "چۆنی؟"


def test_join_empty_list():
    assert sorani_join([]) == ""


def test_join_single_token():
    assert sorani_join(["باشە"]) == "باشە"


# ---------------------------------------------------------------------------
# Round-trip: tokenize then join
# ---------------------------------------------------------------------------

def test_roundtrip_simple():
    text = "ئەم پرۆژەیە باشە."
    tokens = sorani_tokenize(text)
    rejoined = sorani_join(tokens)
    assert rejoined == "ئەم پرۆژەیە باشە."


if __name__ == "__main__":
    test_tokenize_basic_sorani()
    test_tokenize_strips_zwj()
    test_tokenize_preserves_zwnj()
    test_tokenize_splits_conjunctive_waw()
    test_tokenize_standalone_waw_untouched()
    test_tokenize_punctuation_separated()
    test_tokenize_mixed_scripts()
    test_tokenize_empty_string()
    test_tokenize_whitespace_only()
    test_tokenize_extended_arabic_indic_digits()
    test_tokenize_western_digits()
    test_word_tokenize_basic()
    test_word_tokenize_strips_zwj()
    test_word_tokenize_empty()
    test_word_tokenize_keeps_punctuation_attached()
    test_join_basic()
    test_join_suppresses_space_before_punctuation()
    test_join_question_mark()
    test_join_empty_list()
    test_join_single_token()
    test_roundtrip_simple()
    print("All tokenize tests passed!")
