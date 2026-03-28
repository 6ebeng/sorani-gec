"""
Tests for the Sorani Kurdish lexicon module.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.morphology.lexicon import SoraniLexicon, LexiconEntry
from src.morphology.lexicon_parser import AhmadiLexiconParser


def test_lexicon_init():
    """SoraniLexicon initializes and loads dictionary if available."""
    lex = SoraniLexicon()
    assert lex is not None
    print(f"  SoraniLexicon: available={lex.available}, words={len(lex.words)}")


def test_lexicon_available_flag():
    """available flag reflects whether dictionary loaded."""
    lex = SoraniLexicon()
    assert isinstance(lex.available, bool)
    if lex.available:
        assert len(lex.words) > 0


def test_lexicon_unavailable_with_bad_path():
    """SoraniLexicon with bad path sets available=False."""
    lex = SoraniLexicon(dic_path="/nonexistent/dict.dic")
    assert lex.available is False
    assert len(lex.words) == 0


def test_is_correct_known_word():
    """is_correct returns True for common Sorani words."""
    lex = SoraniLexicon()
    if not lex.available:
        print("  SKIP: lexicon not available")
        return
    assert lex.is_correct("من") is True
    print("  'من' is correct")


def test_is_valid_alias():
    """is_valid is an alias for is_correct."""
    lex = SoraniLexicon()
    if not lex.available:
        return
    assert lex.is_valid("من") == lex.is_correct("من")


def test_normalize():
    """normalize returns a string."""
    lex = SoraniLexicon()
    result = lex.normalize("ه‌ەڵۆ")
    assert isinstance(result, str)
    print(f"  normalize('ه‌ەڵۆ') → '{result}'")


def test_suggest_returns_list():
    """suggest returns a list of strings."""
    lex = SoraniLexicon()
    if not lex.available:
        return
    result = lex.suggest("منن")
    assert isinstance(result, list)
    print(f"  suggest('منن') → {result}")


def test_lookup_returns_entries():
    """lookup returns list of LexiconEntry."""
    lex = SoraniLexicon()
    if not lex.available:
        return
    entries = lex.lookup("من")
    assert isinstance(entries, list)
    for e in entries:
        assert isinstance(e, LexiconEntry)
    print(f"  lookup('من') → {len(entries)} entries")


def test_decompose_returns_analyses():
    """decompose returns list of MorphAnalysis objects."""
    lex = SoraniLexicon()
    if not lex.available:
        return
    analyses = lex.decompose("دەچم")
    assert isinstance(analyses, list)
    print(f"  decompose('دەچم') → {len(analyses)} analyses")


def test_get_pos():
    """get_pos returns a POS tag string or None."""
    lex = SoraniLexicon()
    if not lex.available:
        return
    pos = lex.get_pos("من")
    assert pos is None or isinstance(pos, str)
    print(f"  get_pos('من') → {pos}")


def test_verb_stems_returns_dict():
    """verb_stems returns a dict of stems."""
    lex = SoraniLexicon()
    if not lex.available:
        return
    stems = lex.verb_stems()
    assert isinstance(stems, dict)
    print(f"  verb_stems() → {len(stems)} stems")


def test_find_verb_stem():
    """find_verb_stem returns Optional[tuple]."""
    lex = SoraniLexicon()
    if not lex.available:
        return
    result = lex.find_verb_stem("دەچم")
    print(f"  find_verb_stem('دەچم') → {result}")


def test_backward_compat_alias():
    """AhmadiLexiconParser is an alias for SoraniLexicon."""
    assert AhmadiLexiconParser is SoraniLexicon


def test_lexicon_entry_import():
    """LexiconEntry is importable from lexicon_parser."""
    from src.morphology.lexicon_parser import LexiconEntry as LE
    assert LE is LexiconEntry


if __name__ == "__main__":
    print("=== Lexicon Tests ===")
    test_lexicon_init()
    test_lexicon_available_flag()
    test_lexicon_unavailable_with_bad_path()
    test_is_correct_known_word()
    test_is_valid_alias()
    test_normalize()
    test_suggest_returns_list()
    test_lookup_returns_entries()
    test_decompose_returns_analyses()
    test_get_pos()
    test_verb_stems_returns_dict()
    test_find_verb_stem()
    test_backward_compat_alias()
    test_lexicon_entry_import()
    print("All lexicon tests passed!")
