"""
Tests for the synthetic error generation pipeline.
"""

import sys
import os
import tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.errors.pipeline import ErrorPipeline
from src.errors.base import ErrorResult


def test_pipeline_init():
    """Pipeline initializes with all 14 generators."""
    pipeline = ErrorPipeline(error_rate=0.15, seed=42)
    assert len(pipeline.generators) == 14
    print(f"  Pipeline initialized with {len(pipeline.generators)} generators")


def test_pipeline_process_single_sentence():
    """Process a single sentence."""
    pipeline = ErrorPipeline(error_rate=0.5, seed=42)
    result = pipeline.process_sentence("من دەچم بۆ قوتابخانە")
    assert isinstance(result, ErrorResult)
    assert result.original == "من دەچم بۆ قوتابخانە"
    assert isinstance(result.corrupted, str)
    assert isinstance(result.errors, list)
    print(f"  Original:  {result.original}")
    print(f"  Corrupted: {result.corrupted}")
    print(f"  Errors:    {len(result.errors)}")


def test_pipeline_process_preserves_clean_when_no_errors():
    """If no errors are injected, corrupted == original."""
    pipeline = ErrorPipeline(error_rate=0.0, seed=42)  # 0% error rate
    result = pipeline.process_sentence("من دەچم بۆ قوتابخانە")
    assert result.corrupted == result.original
    assert len(result.errors) == 0
    print("  Zero error rate: no corruption (correct)")


def test_pipeline_error_result_to_dict():
    """ErrorResult.to_dict() works."""
    pipeline = ErrorPipeline(error_rate=1.0, seed=42)
    result = pipeline.process_sentence("من دەچم بۆ بازاڕ")
    d = result.to_dict()
    assert "original" in d
    assert "corrupted" in d
    assert "errors" in d
    assert isinstance(d["errors"], list)
    print(f"  to_dict keys: {list(d.keys())}")


def test_pipeline_process_corpus():
    """Full corpus processing pipeline."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a small test input
        input_file = os.path.join(tmpdir, "clean.txt")
        with open(input_file, "w", encoding="utf-8") as f:
            sentences = [
                "من دەچم بۆ قوتابخانە",
                "ئەوان دەنوسن بۆ مامۆستاکانیان",
                "تۆ دەزانیت ئەم بابەتە",
                "ئێمە دەچین بۆ ماڵەوە",
                "ئەو کتێبەکەی خوێندەوە",
            ]
            for s in sentences:
                f.write(s + "\n")

        output_dir = os.path.join(tmpdir, "output")
        pipeline = ErrorPipeline(error_rate=0.5, seed=42)

        stats = pipeline.process_corpus(
            input_file=input_file,
            output_dir=output_dir,
            target_pairs=5,
            corruption_ratio=0.7,
        )

        # Check output files exist
        assert os.path.exists(os.path.join(output_dir, "train.src"))
        assert os.path.exists(os.path.join(output_dir, "train.tgt"))
        assert os.path.exists(os.path.join(output_dir, "annotations.jsonl"))
        assert os.path.exists(os.path.join(output_dir, "generation_stats.json"))

        # Check stats
        assert stats["total"] == 5
        assert stats["corrupted"] + stats["clean_pairs"] == 5

        # Check file line counts match
        with open(os.path.join(output_dir, "train.src"), "r", encoding="utf-8") as f:
            src_lines = f.readlines()
        with open(os.path.join(output_dir, "train.tgt"), "r", encoding="utf-8") as f:
            tgt_lines = f.readlines()
        assert len(src_lines) == len(tgt_lines) == 5

        print(f"  Corpus stats: {stats}")
        print(f"  Output files: src={len(src_lines)} lines, tgt={len(tgt_lines)} lines")


def test_pipeline_oversampling():
    """Pipeline oversamples when input < target."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = os.path.join(tmpdir, "clean.txt")
        with open(input_file, "w", encoding="utf-8") as f:
            f.write("من دەچم بۆ قوتابخانە\n")
            f.write("تۆ دەزانیت ئەم بابەتە\n")

        output_dir = os.path.join(tmpdir, "output")
        pipeline = ErrorPipeline(error_rate=0.3, seed=42)

        stats = pipeline.process_corpus(
            input_file=input_file,
            output_dir=output_dir,
            target_pairs=10,  # more than input
        )

        assert stats["total"] == 10  # oversampled to reach target
        print(f"  Oversampled: 2 input → {stats['total']} pairs")


if __name__ == "__main__":
    print("=== Pipeline Tests ===")
    test_pipeline_init()
    test_pipeline_process_single_sentence()
    test_pipeline_process_preserves_clean_when_no_errors()
    test_pipeline_error_result_to_dict()
    test_pipeline_process_corpus()
    test_pipeline_oversampling()
    print("\nAll pipeline tests passed!")
