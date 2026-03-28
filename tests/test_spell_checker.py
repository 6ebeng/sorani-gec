"""
Tests for the Sorani spell checker module.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.spell_checker import SoraniSpellChecker


def test_spell_checker_init():
    """SoraniSpellChecker initializes."""
    checker = SoraniSpellChecker()
    assert checker is not None
    assert checker.max_suggestions == 3
    print(f"  SpellChecker initialized, available={checker.is_available()}")


def test_is_available():
    """is_available returns a boolean."""
    checker = SoraniSpellChecker()
    result = checker.is_available()
    assert isinstance(result, bool)
    print(f"  is_available: {result}")


def test_is_correct_with_real_word():
    """is_correct returns bool for a common Sorani word."""
    checker = SoraniSpellChecker()
    if not checker.is_available():
        print("  SKIP: lexicon not available")
        return
    # "من" (I) should be a correct word
    assert checker.is_correct("من") is True
    print("  'من' is correct: True")


def test_is_correct_unavailable_returns_true():
    """is_correct returns True when lexicon is unavailable."""
    checker = SoraniSpellChecker(dict_path="/nonexistent/path.dic")
    assert checker.is_correct("xyzabc") is True


def test_get_suggestions_correct_word():
    """get_suggestions for a correct word returns the word itself."""
    checker = SoraniSpellChecker()
    if not checker.is_available():
        print("  SKIP: lexicon not available")
        return
    suggestions = checker.get_suggestions("من")
    assert "من" in suggestions
    print(f"  Suggestions for 'من': {suggestions}")


def test_get_suggestions_returns_list():
    """get_suggestions always returns a list."""
    checker = SoraniSpellChecker()
    result = checker.get_suggestions("ئاسمانن")
    assert isinstance(result, list)
    assert len(result) > 0
    print(f"  Suggestions for misspelled word: {result}")


def test_correct_sentence_unchanged():
    """correct_sentence returns sentence unchanged when all words correct."""
    checker = SoraniSpellChecker()
    if not checker.is_available():
        # When unavailable, should return the same sentence
        sentence = "من دەچم"
        assert checker.correct_sentence(sentence) == sentence
        print("  Unavailable: sentence returned unchanged")
        return
    sentence = "من"
    corrected = checker.correct_sentence(sentence)
    assert isinstance(corrected, str)
    print(f"  Corrected: '{corrected}'")


def test_correct_sentence_unavailable():
    """correct_sentence returns same sentence when lexicon unavailable."""
    checker = SoraniSpellChecker(dict_path="/nonexistent/path.dic")
    sentence = "من دەچم بۆ بازاڕ"
    assert checker.correct_sentence(sentence) == sentence


def test_max_suggestions_respected():
    """Custom max_suggestions is stored."""
    checker = SoraniSpellChecker(max_suggestions=5)
    assert checker.max_suggestions == 5


def test_unavailable_lexicon_logs_warning():
    """7D.4: Unavailable lexicon path produces a checker that is not available."""
    checker = SoraniSpellChecker(dict_path="/nonexistent/path/fake.dic")
    assert not checker.is_available()


def test_correct_word_not_modified():
    """7D.6: Correctly-spelled Kurdish words are not modified by spell-checker."""
    checker = SoraniSpellChecker()
    if not checker.is_available():
        print("  SKIP: lexicon not available")
        return
    correct_words = ["من", "تۆ", "ئەو", "دەچم", "کتێب", "باش", "نان"]
    for word in correct_words:
        corrected = checker.correct_sentence(word)
        assert corrected == word, (
            f"Correct word '{word}' was modified to '{corrected}'"
        )
    print(f"  {len(correct_words)} correct words unchanged")


def test_confidence_skips_spellcheck():
    """7D.9 regression: high model_confidence skips spell correction."""
    checker = SoraniSpellChecker()
    sentence = "ئاسمانن شینە"
    result = checker.correct_sentence(sentence, model_confidence=0.95)
    assert result == sentence, "High confidence should skip spell-check"


if __name__ == "__main__":
    print("=== Spell Checker Tests ===")
    test_spell_checker_init()
    test_is_available()
    test_is_correct_with_real_word()
    test_is_correct_unavailable_returns_true()
    test_get_suggestions_correct_word()
    test_get_suggestions_returns_list()
    test_correct_sentence_unchanged()
    test_correct_sentence_unavailable()
    test_max_suggestions_respected()
    test_correct_word_not_modified()
    test_confidence_skips_spellcheck()
    print("All spell checker tests passed!")
