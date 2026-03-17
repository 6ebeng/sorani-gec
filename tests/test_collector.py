"""
Tests for the data collector module.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.collector import CorpusCollector


def test_collector_init():
    """Collector initializes and creates output directory."""
    collector = CorpusCollector(output_dir="data/raw")
    assert collector.output_dir.exists()
    assert collector.stats["total_sentences"] == 0
    print("  Collector initialized")


def test_is_sorani():
    """Sorani detection works on Kurdish text."""
    # Sorani Kurdish text
    assert CorpusCollector._is_sorani("من دەچم بۆ قوتابخانە لە شاری هەولێر") is True
    # English text
    assert CorpusCollector._is_sorani("This is an English sentence") is False
    # Empty
    assert CorpusCollector._is_sorani("") is False
    # Mixed but mostly Arabic script — majority Kurdish
    assert CorpusCollector._is_sorani("ئەم تێکستە لە زمانی کوردی نووسراوە") is True
    # Pure Arabic (no Kurdish-specific chars like ڕ, ڵ, ڤ, ۆ, ێ, پ, چ, گ)
    assert CorpusCollector._is_sorani("هذا نص باللغة العربية فقط") is False
    print("  Sorani detection: PASSED")


def test_save_stats(tmp_path=None):
    """Stats file is saved correctly."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        collector = CorpusCollector(output_dir=tmpdir)
        collector.stats["sources"]["test"] = 100
        collector.stats["total_sentences"] = 100
        collector.save_stats()

        stats_file = os.path.join(tmpdir, "collection_stats.json")
        assert os.path.exists(stats_file)

        import json
        with open(stats_file, "r", encoding="utf-8") as f:
            saved = json.load(f)
        assert saved["total_sentences"] == 100
        assert saved["sources"]["test"] == 100
    print("  Stats save: PASSED")


if __name__ == "__main__":
    print("=== Corpus Collector Tests ===")
    test_collector_init()
    test_is_sorani()
    test_save_stats()
    print("\nAll collector tests passed!")
