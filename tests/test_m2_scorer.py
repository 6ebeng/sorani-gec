"""
Tests for the M2 format scorer module.
"""

import sys
import os
import tempfile
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.evaluation.m2_scorer import (
    M2Edit, M2Sentence, parse_m2_file, write_m2_file,
    edits_from_sentences, evaluate_m2,
)


def test_m2_edit_creation():
    """M2Edit dataclass creates correctly."""
    edit = M2Edit(start=0, end=1, error_type="NOUN:NUM", correction="کتێبەکان")
    assert edit.start == 0
    assert edit.end == 1
    assert edit.annotator == 0  # default
    print(f"  M2Edit: {edit}")


def test_m2_sentence_creation():
    """M2Sentence stores source and edits."""
    sent = M2Sentence(source="من دەچم")
    assert sent.source == "من دەچم"
    assert sent.edits == []
    sent.edits.append(M2Edit(0, 1, "VERB:FORM", "دەچین"))
    assert len(sent.edits) == 1


def test_write_and_parse_m2_roundtrip(tmp_path):
    """Write M2 file and read it back — roundtrip consistency."""
    sentences = [
        M2Sentence(
            source="من دەچم بۆ بازاڕ",
            edits=[M2Edit(1, 2, "VERB:SVA", "دەچین", annotator=0)],
        ),
        M2Sentence(
            source="تۆ دەنووسیت",
            edits=[
                M2Edit(0, 1, "PRON", "ئەو", annotator=0),
                M2Edit(1, 2, "VERB:SVA", "دەنووسێت", annotator=0),
            ],
        ),
    ]
    out_path = tmp_path / "test.m2"
    write_m2_file(sentences, out_path)
    assert out_path.exists()

    parsed = parse_m2_file(out_path)
    assert len(parsed) == 2
    assert parsed[0].source == "من دەچم بۆ بازاڕ"
    assert len(parsed[0].edits) == 1
    assert parsed[0].edits[0].error_type == "VERB:SVA"
    assert parsed[0].edits[0].correction == "دەچین"
    assert parsed[1].source == "تۆ دەنووسیت"
    assert len(parsed[1].edits) == 2
    print(f"  Roundtrip: {len(parsed)} sentences, edits={[len(s.edits) for s in parsed]}")


def test_parse_m2_file_empty(tmp_path):
    """parse_m2_file on empty file returns empty list."""
    empty = tmp_path / "empty.m2"
    empty.write_text("", encoding="utf-8")
    result = parse_m2_file(empty)
    assert result == []


def test_edits_from_sentences_identical():
    """No edits when hypothesis == reference."""
    hyp_edits, ref_edits = edits_from_sentences(
        "من دەچم", "من دەچم", "من دەچم"
    )
    assert hyp_edits == ref_edits == set()


def test_edits_from_sentences_substitution():
    """Detect a word substitution."""
    hyp_edits, ref_edits = edits_from_sentences(
        "من دەچم", "من دەچین", "من دەچین"
    )
    # Both hyp and ref have same edit (substitution at position 1)
    assert hyp_edits == ref_edits
    assert len(hyp_edits) == 1
    print(f"  Edits: {hyp_edits}")


def test_edits_from_sentences_mismatch():
    """Different corrections produce different edit sets."""
    hyp_edits, ref_edits = edits_from_sentences(
        "من دەچم بۆ", "من دەچین بۆ", "من دەچم بۆ"
    )
    # hyp changed word 1; ref left it as-is
    assert hyp_edits != ref_edits


def test_evaluate_m2_perfect():
    """Perfect corrections → P=1, R=1, F0.5=1."""
    sources = ["من دەچم"]
    hyps = ["من دەچین"]  # same as reference
    refs = ["من دەچین"]
    metrics = evaluate_m2(sources, hyps, refs)
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f05 == 1.0
    print(f"  Perfect: P={metrics.precision}, R={metrics.recall}, F0.5={metrics.f05}")


def test_evaluate_m2_no_edits():
    """No edits needed, no edits made → P=0, R=0 (no TP/FP/FN)."""
    metrics = evaluate_m2(["من دەچم"], ["من دەچم"], ["من دەچم"])
    # All zeros — no edits
    assert metrics.tp == 0
    assert metrics.fp == 0
    assert metrics.fn == 0


def test_evaluate_m2_false_positive():
    """Hypothesis makes edit, reference doesn't → FP only."""
    metrics = evaluate_m2(
        ["من دەچم"], ["من دەچین"], ["من دەچم"]
    )
    assert metrics.fp > 0
    assert metrics.tp == 0
    assert metrics.precision == 0.0


def test_evaluate_m2_false_negative():
    """Reference has edit, hypothesis doesn't → FN only."""
    metrics = evaluate_m2(
        ["من دەچم"], ["من دەچم"], ["من دەچین"]
    )
    assert metrics.fn > 0
    assert metrics.tp == 0
    assert metrics.recall == 0.0


if __name__ == "__main__":
    import tempfile
    print("=== M2 Scorer Tests ===")
    test_m2_edit_creation()
    test_m2_sentence_creation()
    with tempfile.TemporaryDirectory() as td:
        test_write_and_parse_m2_roundtrip(Path(td))
        test_parse_m2_file_empty(Path(td))
    test_edits_from_sentences_identical()
    test_edits_from_sentences_substitution()
    test_edits_from_sentences_mismatch()
    test_evaluate_m2_perfect()
    test_evaluate_m2_no_edits()
    test_evaluate_m2_false_positive()
    test_evaluate_m2_false_negative()
    print("All M2 scorer tests passed!")
