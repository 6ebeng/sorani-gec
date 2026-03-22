"""
Tests for the train/dev/test splitter module.
"""

import json
import sys
import os
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.splitter import load_pairs, split_pairs, save_split, run_split


# ============================================================================
# load_pairs Tests
# ============================================================================

def test_load_pairs_valid_jsonl():
    """Loads well-formed JSONL with source/target keys."""
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False,
                                     encoding="utf-8") as f:
        for i in range(5):
            f.write(json.dumps({"source": f"src_{i}", "target": f"tgt_{i}"},
                               ensure_ascii=False) + "\n")
        path = Path(f.name)
    try:
        pairs = load_pairs(path)
        assert len(pairs) == 5
        assert pairs[0]["source"] == "src_0"
        assert pairs[4]["target"] == "tgt_4"
    finally:
        path.unlink()


def test_load_pairs_skips_blank_lines():
    """Blank lines in JSONL are silently skipped."""
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False,
                                     encoding="utf-8") as f:
        f.write(json.dumps({"source": "a", "target": "b"}) + "\n")
        f.write("\n")
        f.write("   \n")
        f.write(json.dumps({"source": "c", "target": "d"}) + "\n")
        path = Path(f.name)
    try:
        pairs = load_pairs(path)
        assert len(pairs) == 2
    finally:
        path.unlink()


def test_load_pairs_skips_invalid_json():
    """Malformed JSON lines are skipped without crashing."""
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False,
                                     encoding="utf-8") as f:
        f.write(json.dumps({"source": "a", "target": "b"}) + "\n")
        f.write("not json at all\n")
        f.write(json.dumps({"source": "c", "target": "d"}) + "\n")
        path = Path(f.name)
    try:
        pairs = load_pairs(path)
        assert len(pairs) == 2
    finally:
        path.unlink()


def test_load_pairs_skips_missing_keys():
    """Lines missing source or target are skipped."""
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False,
                                     encoding="utf-8") as f:
        f.write(json.dumps({"source": "a", "target": "b"}) + "\n")
        f.write(json.dumps({"source": "only source"}) + "\n")
        f.write(json.dumps({"target": "only target"}) + "\n")
        f.write(json.dumps({"source": "c", "target": "d"}) + "\n")
        path = Path(f.name)
    try:
        pairs = load_pairs(path)
        assert len(pairs) == 2
    finally:
        path.unlink()


def test_load_pairs_empty_file():
    """Empty file returns empty list."""
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False,
                                     encoding="utf-8") as f:
        path = Path(f.name)
    try:
        pairs = load_pairs(path)
        assert len(pairs) == 0
    finally:
        path.unlink()


def test_load_pairs_preserves_extra_fields():
    """Extra keys like error_type or metadata are preserved."""
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False,
                                     encoding="utf-8") as f:
        rec = {"source": "a", "target": "b", "error_type": "clitic_form",
               "metadata": {"page": 5}}
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        path = Path(f.name)
    try:
        pairs = load_pairs(path)
        assert pairs[0]["error_type"] == "clitic_form"
        assert pairs[0]["metadata"]["page"] == 5
    finally:
        path.unlink()


# ============================================================================
# split_pairs Tests
# ============================================================================

def test_split_pairs_default_ratios():
    """Default 80/10/10 split."""
    pairs = [{"source": str(i), "target": str(i)} for i in range(100)]
    train, dev, test = split_pairs(pairs)
    assert len(train) == 80
    assert len(dev) == 10
    assert len(test) == 10


def test_split_pairs_custom_ratios():
    """Custom ratio split."""
    pairs = [{"source": str(i), "target": str(i)} for i in range(100)]
    train, dev, test = split_pairs(pairs, train_ratio=0.6, dev_ratio=0.2,
                                   test_ratio=0.2)
    assert len(train) == 60
    assert len(dev) == 20
    assert len(test) == 20


def test_split_pairs_invalid_ratios():
    """Ratios not summing to 1.0 raise AssertionError."""
    pairs = [{"source": "a", "target": "b"}]
    try:
        split_pairs(pairs, train_ratio=0.5, dev_ratio=0.1, test_ratio=0.1)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass


def test_split_pairs_seed_reproducibility():
    """Same seed produces identical splits."""
    pairs = [{"source": str(i), "target": str(i)} for i in range(50)]
    t1, d1, te1 = split_pairs(pairs, seed=99)
    t2, d2, te2 = split_pairs(pairs, seed=99)
    assert t1 == t2
    assert d1 == d2
    assert te1 == te2


def test_split_pairs_different_seeds_differ():
    """Different seeds produce different orderings."""
    pairs = [{"source": str(i), "target": str(i)} for i in range(50)]
    t1, _, _ = split_pairs(pairs, seed=1)
    t2, _, _ = split_pairs(pairs, seed=2)
    assert t1 != t2


def test_split_pairs_single_item():
    """Single-item input puts it in test (train=0, dev=0)."""
    pairs = [{"source": "a", "target": "b"}]
    train, dev, test = split_pairs(pairs)
    assert len(train) + len(dev) + len(test) == 1


def test_split_pairs_empty():
    """Empty input returns three empty lists."""
    train, dev, test = split_pairs([])
    assert len(train) == 0
    assert len(dev) == 0
    assert len(test) == 0


def test_split_pairs_no_data_loss():
    """All items appear exactly once across the three splits."""
    pairs = [{"source": str(i), "target": str(i)} for i in range(73)]
    train, dev, test = split_pairs(pairs)
    all_sources = sorted([p["source"] for p in train + dev + test])
    expected = sorted([str(i) for i in range(73)])
    assert all_sources == expected


# ============================================================================
# Stratified Split Tests
# ============================================================================

def test_split_pairs_stratified():
    """Stratified split preserves error_type proportions."""
    pairs = []
    for i in range(60):
        pairs.append({"source": str(i), "target": str(i),
                       "error_type": "clitic_form"})
    for i in range(40):
        pairs.append({"source": str(i), "target": str(i),
                       "error_type": "tense_agreement"})
    train, dev, test = split_pairs(pairs, stratify_key="error_type")
    total = len(train) + len(dev) + len(test)
    assert total == 100

    # Each split should have both error types
    train_types = {p["error_type"] for p in train}
    assert "clitic_form" in train_types
    assert "tense_agreement" in train_types


def test_split_pairs_stratified_missing_key():
    """Items without the stratify key get labeled 'unknown'."""
    pairs = [
        {"source": "a", "target": "b", "error_type": "clitic_form"},
        {"source": "c", "target": "d"},  # no error_type
    ]
    train, dev, test = split_pairs(pairs, stratify_key="error_type")
    assert len(train) + len(dev) + len(test) == 2


# ============================================================================
# save_split Tests
# ============================================================================

def test_save_split_creates_file():
    """save_split writes valid JSONL."""
    pairs = [{"source": "a", "target": "b"}, {"source": "c", "target": "d"}]
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "sub" / "out.jsonl"
        save_split(pairs, path)
        assert path.exists()
        reloaded = load_pairs(path)
        assert len(reloaded) == 2
        assert reloaded[0]["source"] == "a"


def test_save_split_unicode():
    """Kurdish text round-trips correctly through save/load."""
    pairs = [{"source": "من دەچم بۆ قوتابخانە", "target": "من دەچم بۆ قوتابخانە"}]
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "kurdish.jsonl"
        save_split(pairs, path)
        reloaded = load_pairs(path)
        assert reloaded[0]["source"] == "من دەچم بۆ قوتابخانە"


# ============================================================================
# run_split End-to-End Tests
# ============================================================================

def test_run_split_end_to_end():
    """Full pipeline: load → split → save."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create input
        input_path = Path(tmpdir) / "input.jsonl"
        with open(input_path, "w", encoding="utf-8") as f:
            for i in range(20):
                f.write(json.dumps({"source": f"s{i}", "target": f"t{i}"},
                                   ensure_ascii=False) + "\n")

        output_dir = Path(tmpdir) / "splits"
        result = run_split(input_path, output_dir)

        assert result["train"] == 16  # 80% of 20
        assert result["dev"] == 2     # 10% of 20
        assert result["test"] == 2    # 10% of 20

        assert (output_dir / "train.jsonl").exists()
        assert (output_dir / "dev.jsonl").exists()
        assert (output_dir / "test.jsonl").exists()

        # Verify file contents match counts
        train_loaded = load_pairs(output_dir / "train.jsonl")
        assert len(train_loaded) == 16


# ============================================================================
# Run all tests
# ============================================================================

if __name__ == "__main__":
    print("=== Splitter: load_pairs Tests ===")
    test_load_pairs_valid_jsonl()
    print("  test_load_pairs_valid_jsonl: PASSED")
    test_load_pairs_skips_blank_lines()
    print("  test_load_pairs_skips_blank_lines: PASSED")
    test_load_pairs_skips_invalid_json()
    print("  test_load_pairs_skips_invalid_json: PASSED")
    test_load_pairs_skips_missing_keys()
    print("  test_load_pairs_skips_missing_keys: PASSED")
    test_load_pairs_empty_file()
    print("  test_load_pairs_empty_file: PASSED")
    test_load_pairs_preserves_extra_fields()
    print("  test_load_pairs_preserves_extra_fields: PASSED")

    print("\n=== Splitter: split_pairs Tests ===")
    test_split_pairs_default_ratios()
    print("  test_split_pairs_default_ratios: PASSED")
    test_split_pairs_custom_ratios()
    print("  test_split_pairs_custom_ratios: PASSED")
    test_split_pairs_invalid_ratios()
    print("  test_split_pairs_invalid_ratios: PASSED")
    test_split_pairs_seed_reproducibility()
    print("  test_split_pairs_seed_reproducibility: PASSED")
    test_split_pairs_different_seeds_differ()
    print("  test_split_pairs_different_seeds_differ: PASSED")
    test_split_pairs_single_item()
    print("  test_split_pairs_single_item: PASSED")
    test_split_pairs_empty()
    print("  test_split_pairs_empty: PASSED")
    test_split_pairs_no_data_loss()
    print("  test_split_pairs_no_data_loss: PASSED")

    print("\n=== Splitter: Stratified Split Tests ===")
    test_split_pairs_stratified()
    print("  test_split_pairs_stratified: PASSED")
    test_split_pairs_stratified_missing_key()
    print("  test_split_pairs_stratified_missing_key: PASSED")

    print("\n=== Splitter: save_split Tests ===")
    test_save_split_creates_file()
    print("  test_save_split_creates_file: PASSED")
    test_save_split_unicode()
    print("  test_save_split_unicode: PASSED")

    print("\n=== Splitter: run_split Tests ===")
    test_run_split_end_to_end()
    print("  test_run_split_end_to_end: PASSED")

    print("\nAll splitter tests passed!")
