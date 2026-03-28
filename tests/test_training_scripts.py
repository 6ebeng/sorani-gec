"""
Tests for training script data loading functions.
"""

import sys
import os
import json
import importlib.util
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_SCRIPT_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts")


def test_baseline_load_pairs_jsonl(tmp_path):
    """load_pairs reads .jsonl format."""
    # Create a JSONL file
    data = [
        {"source": "من دەچم", "target": "من دەچم"},
        {"source": "تۆ دەزانیت", "target": "تۆ دەزانیت"},
    ]
    jl = tmp_path / "train.jsonl"
    with open(jl, "w", encoding="utf-8") as f:
        for rec in data:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Import load_pairs from the script
    script_path = os.path.join(_SCRIPT_DIR, "05_train_baseline.py")
    if not os.path.exists(script_path):
        print("  SKIP: 05_train_baseline.py not found")
        return
    spec = importlib.util.spec_from_file_location("train_baseline", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    sources, targets = mod.load_pairs(jl)
    assert len(sources) == 2
    assert len(targets) == 2
    assert sources[0] == "من دەچم"
    assert targets[1] == "تۆ دەزانیت"
    print(f"  load_pairs(.jsonl): {len(sources)} pairs")


def test_baseline_load_pairs_src_tgt(tmp_path):
    """load_pairs reads .src/.tgt format."""
    (tmp_path / "train.src").write_text("من دەچم\nتۆ\n", encoding="utf-8")
    (tmp_path / "train.tgt").write_text("من دەچم\nتۆ دەزانیت\n", encoding="utf-8")

    script_path = os.path.join(_SCRIPT_DIR, "05_train_baseline.py")
    if not os.path.exists(script_path):
        print("  SKIP: 05_train_baseline.py not found")
        return
    spec = importlib.util.spec_from_file_location("train_baseline2", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Pass a path that has .src/.tgt siblings (use stem "train")
    sources, targets = mod.load_pairs(tmp_path / "train.txt")
    assert len(sources) == 2
    assert len(targets) == 2
    print(f"  load_pairs(.src/.tgt): {len(sources)} pairs")


def test_morphaware_load_jsonl(tmp_path):
    """load_jsonl reads source/target/metadata from JSONL."""
    data = [
        {"source": "s1", "target": "t1", "error_type": "subject_verb"},
        {"source": "s2", "target": "t2", "error_type": "clitic"},
    ]
    jl = tmp_path / "train.jsonl"
    with open(jl, "w", encoding="utf-8") as f:
        for rec in data:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    script_path = os.path.join(_SCRIPT_DIR, "06_train_morphaware.py")
    if not os.path.exists(script_path):
        print("  SKIP: 06_train_morphaware.py not found")
        return
    spec = importlib.util.spec_from_file_location("train_morphaware", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    sources, targets, records = mod.load_jsonl(jl)
    assert len(sources) == 2
    assert len(targets) == 2
    assert len(records) == 2
    assert records[0]["error_type"] == "subject_verb"
    print(f"  load_jsonl: {len(sources)} sources, {len(records)} records")


def test_morphaware_load_plain_pairs(tmp_path):
    """load_plain_pairs reads train.src + train.tgt."""
    (tmp_path / "train.src").write_text("a\nb\nc\n", encoding="utf-8")
    (tmp_path / "train.tgt").write_text("x\ny\nz\n", encoding="utf-8")

    script_path = os.path.join(_SCRIPT_DIR, "06_train_morphaware.py")
    if not os.path.exists(script_path):
        print("  SKIP: 06_train_morphaware.py not found")
        return
    spec = importlib.util.spec_from_file_location("train_morphaware2", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    sources, targets = mod.load_plain_pairs(tmp_path)
    assert sources == ["a", "b", "c"]
    assert targets == ["x", "y", "z"]
    print(f"  load_plain_pairs: {len(sources)} pairs")


def test_morphaware_load_jsonl_empty_lines(tmp_path):
    """load_jsonl skips empty lines."""
    jl = tmp_path / "train.jsonl"
    jl.write_text(
        '{"source":"a","target":"b"}\n\n{"source":"c","target":"d"}\n',
        encoding="utf-8",
    )

    script_path = os.path.join(_SCRIPT_DIR, "06_train_morphaware.py")
    if not os.path.exists(script_path):
        return
    spec = importlib.util.spec_from_file_location("train_morphaware3", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    sources, targets, records = mod.load_jsonl(jl)
    assert len(sources) == 2


if __name__ == "__main__":
    import tempfile
    print("=== Training Script Tests ===")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        test_baseline_load_pairs_jsonl(td)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        test_baseline_load_pairs_src_tgt(td)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        test_morphaware_load_jsonl(td)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        test_morphaware_load_plain_pairs(td)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        test_morphaware_load_jsonl_empty_lines(td)
    print("All training script tests passed!")
