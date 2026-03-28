"""Tests for the GLEU scorer."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.evaluation.gleu_scorer import compute_gleu, compute_gleu_per_sentence


def test_gleu_perfect_correction():
    """Perfect correction should produce a high GLEU score."""
    sources =    ["من دەچین بۆ قوتابخانە"]
    hypotheses = ["من دەچم بۆ قوتابخانە"]
    references = ["من دەچم بۆ قوتابخانە"]

    score = compute_gleu(sources, hypotheses, references)
    assert score > 0.5, f"Perfect correction should yield high GLEU, got {score:.4f}"


def test_gleu_no_change_needed():
    """If source already equals reference and hypothesis matches, GLEU should be high."""
    sources =    ["من دەچم بۆ قوتابخانە"]
    hypotheses = ["من دەچم بۆ قوتابخانە"]
    references = ["من دەچم بۆ قوتابخانە"]

    score = compute_gleu(sources, hypotheses, references)
    assert score > 0.8, f"No correction needed, GLEU should be near 1.0, got {score:.4f}"


def test_gleu_wrong_correction():
    """Hypothesis introduces a new error — GLEU should be low."""
    sources =    ["من دەچین بۆ قوتابخانە"]
    hypotheses = ["تۆ دەچین بۆ قوتابخانە"]   # wrong pronoun swap, didn't fix verb
    references = ["من دەچم بۆ قوتابخانە"]

    score = compute_gleu(sources, hypotheses, references)
    # Should be quite low
    assert score < 0.7, f"Wrong correction GLEU should be low, got {score:.4f}"


def test_gleu_empty_inputs():
    """Empty inputs should return 0."""
    score = compute_gleu([""], [""], [""])
    assert score == 0.0


def test_gleu_corpus_level_averaging():
    """Corpus-level GLEU averages across sentences."""
    sources =    ["من دەچین", "ئەو خوێندن"]
    hypotheses = ["من دەچم",  "ئەو خوێندن"]   # first corrected, second unchanged
    references = ["من دەچم",  "ئەو دەخوێنێ"]  # second needs correction

    score = compute_gleu(sources, hypotheses, references)
    assert 0.0 < score < 1.0, f"Mixed results GLEU should be between 0 and 1, got {score:.4f}"


def test_gleu_per_sentence():
    """Per-sentence GLEU returns a list with one score per sentence."""
    sources =    ["من دەچین", "ئەو خوێندن"]
    hypotheses = ["من دەچم",  "ئەو خوێندن"]
    references = ["من دەچم",  "ئەو دەخوێنێ"]

    scores = compute_gleu_per_sentence(sources, hypotheses, references)
    assert len(scores) == 2
    assert scores[0] > scores[1], "First sentence (correct) should score higher"


def test_gleu_symmetry_with_identical():
    """When hyp == ref, score should be maximal regardless of source."""
    sources =    ["ئەو کتێبەکان خوێندمەوە"]
    hypotheses = ["ئەو کتێبانە خوێندمەوە"]
    references = ["ئەو کتێبانە خوێندمەوە"]

    score = compute_gleu(sources, hypotheses, references)
    assert score > 0.5, f"Exact match to reference should be high, got {score:.4f}"


def test_gleu_single_word():
    """Single-word sentences should work without errors."""
    score = compute_gleu(["دەچم"], ["دەچم"], ["دەچم"])
    assert score > 0.0
