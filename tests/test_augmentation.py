"""
Tests for data augmentation module.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.augmentation import SoraniAugmenter


def test_augmenter_init():
    """SoraniAugmenter initializes with a seed."""
    aug = SoraniAugmenter(seed=42)
    assert aug is not None
    print("  Augmenter initialized with seed=42")


def test_random_swap_basic():
    """random_swap swaps adjacent words."""
    aug = SoraniAugmenter(seed=42)
    sentence = "من دەچم بۆ قوتابخانە"
    swapped = aug.random_swap(sentence, n_swaps=1)
    words_orig = sentence.split()
    words_swap = swapped.split()
    assert len(words_orig) == len(words_swap), "Word count should stay the same"
    assert set(words_orig) == set(words_swap), "Same words, just reordered"
    print(f"  Original: {sentence}")
    print(f"  Swapped:  {swapped}")


def test_random_swap_single_word():
    """random_swap on single-word sentence returns unchanged."""
    aug = SoraniAugmenter(seed=42)
    assert aug.random_swap("سڵاو") == "سڵاو"


def test_random_swap_two_words():
    """random_swap on two-word sentence swaps the pair."""
    aug = SoraniAugmenter(seed=42)
    result = aug.random_swap("من دەچم", n_swaps=1)
    assert result in ("من دەچم", "دەچم من")


def test_random_deletion_basic():
    """random_deletion removes some words."""
    aug = SoraniAugmenter(seed=42)
    sentence = "من دەچم بۆ قوتابخانە لە ئێستادا"
    deleted = aug.random_deletion(sentence, delete_prob=0.5)
    assert len(deleted.split()) <= len(sentence.split())
    assert len(deleted) > 0, "Should retain at least one word"
    print(f"  Original: {sentence}")
    print(f"  Deleted:  {deleted}")


def test_random_deletion_single_word():
    """random_deletion on single word returns it unchanged."""
    aug = SoraniAugmenter(seed=42)
    assert aug.random_deletion("سڵاو") == "سڵاو"


def test_random_deletion_zero_prob():
    """random_deletion with delete_prob=0 keeps all words."""
    aug = SoraniAugmenter(seed=42)
    sentence = "من دەچم بۆ قوتابخانە"
    assert aug.random_deletion(sentence, delete_prob=0.0) == sentence


def test_augment_pair_swap():
    """augment_pair with swap strategy augments source only."""
    aug = SoraniAugmenter(seed=42)
    src, tgt = aug.augment_pair("من دەچم بۆ بازاڕ", "من دەچم بۆ بازاڕ", strategy="swap")
    assert tgt == "من دەچم بۆ بازاڕ", "Target should remain unchanged for swap"
    print(f"  Swap src: {src}")


def test_augment_pair_delete():
    """augment_pair with delete strategy augments source only."""
    aug = SoraniAugmenter(seed=42)
    src, tgt = aug.augment_pair(
        "من دەچم بۆ بازاڕ", "من دەچم بۆ بازاڕ",
        strategy="delete", delete_prob=0.5,
    )
    assert tgt == "من دەچم بۆ بازاڕ", "Target should remain unchanged for delete"


def test_augment_pair_unknown_strategy():
    """augment_pair with unknown strategy returns pair unchanged."""
    aug = SoraniAugmenter(seed=42)
    src, tgt = aug.augment_pair("a b c", "a b c", strategy="unknown_xyz")
    assert src == "a b c"
    assert tgt == "a b c"


def test_augment_corpus_basic():
    """augment_corpus adds augmented pairs to corpus."""
    aug = SoraniAugmenter(seed=42)
    pairs = [
        {"source": "من دەچم بۆ بازاڕ", "target": "من دەچم بۆ بازاڕ"},
        {"source": "تۆ دەزانیت", "target": "تۆ دەزانیت"},
    ]
    result = aug.augment_corpus(pairs, strategies=["swap", "delete"], augment_ratio=1.0)
    assert len(result) >= len(pairs), "Should have originals + augmented pairs"
    # All items should have source/target keys
    for item in result:
        assert "source" in item
        assert "target" in item
    print(f"  Input: {len(pairs)} pairs → Output: {len(result)} pairs")


def test_augment_corpus_zero_ratio():
    """augment_corpus with ratio=0 returns only originals."""
    aug = SoraniAugmenter(seed=42)
    pairs = [{"source": "a b", "target": "a b"}]
    result = aug.augment_corpus(pairs, strategies=["swap"], augment_ratio=0.0)
    assert len(result) == 1


def test_deterministic_with_seed():
    """Same seed produces same results."""
    aug1 = SoraniAugmenter(seed=123)
    aug2 = SoraniAugmenter(seed=123)
    sentence = "من دەچم بۆ قوتابخانە لەسەر ئه‌م بابەتە"
    assert aug1.random_swap(sentence) == aug2.random_swap(sentence)
    assert aug1.random_deletion(sentence) == aug2.random_deletion(sentence)


if __name__ == "__main__":
    print("=== Augmentation Tests ===")
    test_augmenter_init()
    test_random_swap_basic()
    test_random_swap_single_word()
    test_random_swap_two_words()
    test_random_deletion_basic()
    test_random_deletion_single_word()
    test_random_deletion_zero_prob()
    test_augment_pair_swap()
    test_augment_pair_delete()
    test_augment_pair_unknown_strategy()
    test_augment_corpus_basic()
    test_augment_corpus_zero_ratio()
    test_deterministic_with_seed()
    print("All augmentation tests passed!")
