"""
Tests for the corpus catalog module — category-aware corpus management.
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.corpus_catalog import (
    CATEGORIES,
    CorpusCatalog,
    SourceDocument,
    CatalogStats,
    KTC_CATEGORY_MAP,
    _CATEGORY_HINTS,
)


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

def _make_corpus(tmpdir: str, files: dict[str, list[str]]) -> str:
    """Create text files in tmpdir. files maps filename→list of sentences."""
    for fname, sentences in files.items():
        fpath = os.path.join(tmpdir, fname)
        with open(fpath, "w", encoding="utf-8") as f:
            for s in sentences:
                f.write(s + "\n")
    return tmpdir


def _make_catalog_json(tmpdir: str, documents: list[dict]) -> str:
    """Write a catalog JSON file and return its path."""
    path = os.path.join(tmpdir, "catalog.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"documents": documents}, f, ensure_ascii=False)
    return path


# ---------------------------------------------------------------
# Category system
# ---------------------------------------------------------------

def test_categories_not_empty():
    """CATEGORIES list is populated."""
    assert len(CATEGORIES) >= 10
    assert "linguistics" in CATEGORIES
    assert "general" in CATEGORIES


def test_category_hints_cover_all_categories():
    """Every non-general category has keyword hints."""
    for cat in CATEGORIES:
        if cat == "general":
            continue
        assert cat in _CATEGORY_HINTS, f"Missing hints for {cat}"
        assert len(_CATEGORY_HINTS[cat]) > 0


# ---------------------------------------------------------------
# Category inference
# ---------------------------------------------------------------

def test_infer_category_linguistics():
    assert CorpusCatalog._infer_category("morphology_thesis.txt") == "linguistics"


def test_infer_category_history():
    assert CorpusCatalog._infer_category("historical_kurdistan.txt") == "history"


def test_infer_category_unknown_falls_to_general():
    assert CorpusCatalog._infer_category("random_file_12345.txt") == "general"


def test_infer_category_kurdish_keywords():
    # Kurdish keyword for education: پەروەردە
    assert CorpusCatalog._infer_category("پەروەردە_thesis.txt") == "education"


# ---------------------------------------------------------------
# Catalog loading — explicit JSON
# ---------------------------------------------------------------

def test_load_catalog_json():
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_corpus(tmpdir, {
            "ling.txt": ["sentence one", "sentence two"],
            "hist.txt": ["sentence three"],
        })
        cat_path = _make_catalog_json(tmpdir, [
            {"filename": "ling.txt", "category": "linguistics"},
            {"filename": "hist.txt", "category": "history"},
        ])

        catalog = CorpusCatalog(tmpdir, catalog_path=cat_path)
        assert len(catalog.documents) == 2
        assert catalog.documents[0].category == "linguistics"
        assert catalog.documents[1].category == "history"


def test_load_catalog_unknown_category_falls_to_general():
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_corpus(tmpdir, {"f.txt": ["a"]})
        cat_path = _make_catalog_json(tmpdir, [
            {"filename": "f.txt", "category": "nonexistent_category"},
        ])
        catalog = CorpusCatalog(tmpdir, catalog_path=cat_path)
        assert catalog.documents[0].category == "general"


# ---------------------------------------------------------------
# Auto-cataloging from filenames
# ---------------------------------------------------------------

def test_auto_catalog():
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_corpus(tmpdir, {
            "morphology_analysis.txt": ["a", "b"],
            "historical_review.txt": ["c"],
            "random_notes.txt": ["d"],
        })
        catalog = CorpusCatalog(tmpdir)
        assert len(catalog.documents) == 3

        cats = {d.filename: d.category for d in catalog.documents}
        assert cats["morphology_analysis.txt"] == "linguistics"
        assert cats["historical_review.txt"] == "history"
        assert cats["random_notes.txt"] == "general"


# ---------------------------------------------------------------
# Sentence loading
# ---------------------------------------------------------------

def test_load_sentences_counts():
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_corpus(tmpdir, {
            "ling1.txt": ["sent1", "sent2", "sent3"],
            "hist1.txt": ["sent4", "sent5"],
        })
        cat_path = _make_catalog_json(tmpdir, [
            {"filename": "ling1.txt", "category": "linguistics"},
            {"filename": "hist1.txt", "category": "history"},
        ])
        catalog = CorpusCatalog(tmpdir, catalog_path=cat_path)
        stats = catalog.load_sentences()

        assert stats.total_sentences == 5
        assert stats.total_documents == 2
        assert stats.per_category["linguistics"] == 3
        assert stats.per_category["history"] == 2


def test_load_sentences_empty_lines_skipped():
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_corpus(tmpdir, {
            "f.txt": ["good line", "", "  ", "another good line"],
        })
        cat_path = _make_catalog_json(tmpdir, [
            {"filename": "f.txt", "category": "general"},
        ])
        catalog = CorpusCatalog(tmpdir, catalog_path=cat_path)
        stats = catalog.load_sentences()
        assert stats.total_sentences == 2


def test_load_sentences_missing_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        cat_path = _make_catalog_json(tmpdir, [
            {"filename": "ghost.txt", "category": "linguistics"},
        ])
        catalog = CorpusCatalog(tmpdir, catalog_path=cat_path)
        stats = catalog.load_sentences()
        assert stats.total_sentences == 0


# ---------------------------------------------------------------
# Balanced sampling
# ---------------------------------------------------------------

def test_balanced_sample_even_distribution():
    """When all categories have enough data, output is roughly even."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_corpus(tmpdir, {
            "ling.txt": [f"ling_{i}" for i in range(200)],
            "hist.txt": [f"hist_{i}" for i in range(200)],
            "edu.txt": [f"edu_{i}" for i in range(200)],
        })
        cat_path = _make_catalog_json(tmpdir, [
            {"filename": "ling.txt", "category": "linguistics"},
            {"filename": "hist.txt", "category": "history"},
            {"filename": "edu.txt", "category": "education"},
        ])
        catalog = CorpusCatalog(tmpdir, catalog_path=cat_path)
        catalog.load_sentences()

        sampled = catalog.balanced_sample(target_sentences=150, min_per_category=10)

        assert len(sampled) == 3
        total = sum(len(s) for s in sampled.values())
        assert total == 150

        # Each category should get exactly 50
        for cat in sampled:
            assert len(sampled[cat]) == 50


def test_balanced_sample_small_category_gives_all():
    """A small category contributes everything; deficit goes to larger ones."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_corpus(tmpdir, {
            "big.txt": [f"big_{i}" for i in range(500)],
            "small.txt": [f"small_{i}" for i in range(30)],
        })
        cat_path = _make_catalog_json(tmpdir, [
            {"filename": "big.txt", "category": "linguistics"},
            {"filename": "small.txt", "category": "history"},
        ])
        catalog = CorpusCatalog(tmpdir, catalog_path=cat_path)
        catalog.load_sentences()

        sampled = catalog.balanced_sample(target_sentences=200, min_per_category=10)

        assert len(sampled["history"]) == 30  # all of its sentences
        assert len(sampled["linguistics"]) == 170  # absorbed the deficit
        assert sum(len(s) for s in sampled.values()) == 200


def test_balanced_sample_min_per_category_filters():
    """Categories below min_per_category are excluded."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_corpus(tmpdir, {
            "big.txt": [f"big_{i}" for i in range(300)],
            "tiny.txt": [f"tiny_{i}" for i in range(5)],
        })
        cat_path = _make_catalog_json(tmpdir, [
            {"filename": "big.txt", "category": "linguistics"},
            {"filename": "tiny.txt", "category": "history"},
        ])
        catalog = CorpusCatalog(tmpdir, catalog_path=cat_path)
        catalog.load_sentences()

        sampled = catalog.balanced_sample(target_sentences=100, min_per_category=50)

        assert "history" not in sampled
        assert len(sampled["linguistics"]) == 100


def test_balanced_sample_deterministic():
    """Same seed gives same sample."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_corpus(tmpdir, {
            "a.txt": [f"a_{i}" for i in range(200)],
            "b.txt": [f"b_{i}" for i in range(200)],
        })
        cat_path = _make_catalog_json(tmpdir, [
            {"filename": "a.txt", "category": "linguistics"},
            {"filename": "b.txt", "category": "history"},
        ])

        s1 = CorpusCatalog(tmpdir, catalog_path=cat_path, seed=42)
        s1.load_sentences()
        r1 = s1.balanced_sample(target_sentences=100)

        s2 = CorpusCatalog(tmpdir, catalog_path=cat_path, seed=42)
        s2.load_sentences()
        r2 = s2.balanced_sample(target_sentences=100)

        assert r1["linguistics"] == r2["linguistics"]
        assert r1["history"] == r2["history"]


def test_balanced_sample_no_active_categories():
    """When no category meets the minimum, return empty."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_corpus(tmpdir, {
            "f.txt": [f"s_{i}" for i in range(5)],
        })
        cat_path = _make_catalog_json(tmpdir, [
            {"filename": "f.txt", "category": "linguistics"},
        ])
        catalog = CorpusCatalog(tmpdir, catalog_path=cat_path)
        catalog.load_sentences()

        sampled = catalog.balanced_sample(target_sentences=100, min_per_category=50)
        assert sampled == {}


# ---------------------------------------------------------------
# I/O — save balanced corpus
# ---------------------------------------------------------------

def test_save_balanced_corpus_format():
    """Output file has tab-separated category\\tsentence format."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_corpus(tmpdir, {
            "a.txt": [f"sent_{i}" for i in range(50)],
            "b.txt": [f"sent_{i+50}" for i in range(50)],
        })
        cat_path = _make_catalog_json(tmpdir, [
            {"filename": "a.txt", "category": "linguistics"},
            {"filename": "b.txt", "category": "history"},
        ])
        catalog = CorpusCatalog(tmpdir, catalog_path=cat_path)
        catalog.load_sentences()

        out_path = os.path.join(tmpdir, "balanced.txt")
        stats = catalog.save_balanced_corpus(out_path, target_sentences=60, min_per_category=10)

        assert stats.total_sentences == 60

        with open(out_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) == 60

        # Every line has a tab separator
        for line in lines:
            parts = line.strip().split("\t")
            assert len(parts) == 2
            assert parts[0] in CATEGORIES


def test_save_balanced_corpus_stats_json():
    """Companion _stats.json file is created."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_corpus(tmpdir, {
            "a.txt": [f"s_{i}" for i in range(100)],
        })
        cat_path = _make_catalog_json(tmpdir, [
            {"filename": "a.txt", "category": "linguistics"},
        ])
        catalog = CorpusCatalog(tmpdir, catalog_path=cat_path)
        catalog.load_sentences()

        out_path = os.path.join(tmpdir, "output", "balanced.txt")
        catalog.save_balanced_corpus(out_path, target_sentences=50, min_per_category=10)

        stats_path = os.path.join(tmpdir, "output", "balanced_stats.json")
        assert os.path.exists(stats_path)

        with open(stats_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["actual_sentences"] == 50
        assert "per_category" in data


# ---------------------------------------------------------------
# Catalog save / round-trip
# ---------------------------------------------------------------

def test_save_catalog_roundtrip():
    """Catalog can be saved and reloaded."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_corpus(tmpdir, {
            "a.txt": ["one", "two"],
            "b.txt": ["three"],
        })
        cat_path = _make_catalog_json(tmpdir, [
            {"filename": "a.txt", "category": "linguistics", "title": "Test A"},
            {"filename": "b.txt", "category": "history", "title": "Test B"},
        ])

        c1 = CorpusCatalog(tmpdir, catalog_path=cat_path)
        c1.load_sentences()

        saved_path = os.path.join(tmpdir, "saved_catalog.json")
        c1.save_catalog(saved_path)

        c2 = CorpusCatalog(tmpdir, catalog_path=saved_path)
        assert len(c2.documents) == 2
        assert c2.documents[0].category == "linguistics"


# ---------------------------------------------------------------
# SourceDocument and CatalogStats
# ---------------------------------------------------------------

def test_source_document_defaults():
    doc = SourceDocument(filename="test.txt", category="general")
    assert doc.sentence_count == 0
    assert doc.char_count == 0
    assert doc.title == ""


def test_catalog_stats_defaults():
    stats = CatalogStats()
    assert stats.total_sentences == 0
    assert stats.per_category == {}


# ---------------------------------------------------------------
# KTC category mapping
# ---------------------------------------------------------------

def test_ktc_category_map_not_empty():
    """KTC_CATEGORY_MAP has all 12 KTC directories."""
    assert len(KTC_CATEGORY_MAP) == 12


def test_ktc_category_map_values_are_valid():
    """Every mapped value is in CATEGORIES."""
    for ktc_cat, our_cat in KTC_CATEGORY_MAP.items():
        assert our_cat in CATEGORIES, f"{ktc_cat} maps to unknown {our_cat}"


def test_ktc_category_map_expected_entries():
    """Spot-check specific mappings."""
    assert KTC_CATEGORY_MAP["economy"] == "economics"
    assert KTC_CATEGORY_MAP["physics"] == "sciences"
    assert KTC_CATEGORY_MAP["theology"] == "islamic_studies"
    assert KTC_CATEGORY_MAP["kurdish"] == "linguistics"
    assert KTC_CATEGORY_MAP["history"] == "history"
    assert KTC_CATEGORY_MAP["human-rights"] == "law"


# ---------------------------------------------------------------
# KTC catalog construction (from_ktc)
# ---------------------------------------------------------------

def _make_ktc_tree(tmpdir: str) -> str:
    """Create a minimal fake KTC directory structure."""
    ktc_root = os.path.join(tmpdir, "ktc")
    os.makedirs(ktc_root, exist_ok=True)
    # Create a few KTC category directories with text files
    for ktc_cat, sentences in [
        ("economy", ["جملەی ئابووری ١", "جملەی ئابووری ٢"]),
        ("history", ["جملەی مێژوو ١", "جملەی مێژوو ٢", "جملەی مێژوو ٣"]),
        ("physics", ["جملەی فیزیا ١"]),
        ("kurdish", ["جملەی زمان ١", "جملەی زمان ٢"]),
    ]:
        cat_dir = os.path.join(ktc_root, ktc_cat)
        os.makedirs(cat_dir, exist_ok=True)
        fpath = os.path.join(cat_dir, f"01s-ch01-2018.txt")
        with open(fpath, "w", encoding="utf-8") as f:
            for s in sentences:
                f.write(s + "\n")
    # Kurdish also has a subdirectory
    lit_dir = os.path.join(ktc_root, "kurdish", "Literature")
    os.makedirs(lit_dir, exist_ok=True)
    with open(os.path.join(lit_dir, "07s-ch01-2015.txt"), "w", encoding="utf-8") as f:
        f.write("ئەدەبی کوردی\n")
    return ktc_root


def test_from_ktc_builds_catalog():
    with tempfile.TemporaryDirectory() as tmpdir:
        ktc_root = _make_ktc_tree(tmpdir)
        catalog = CorpusCatalog.from_ktc(ktc_root)
        # 4 directories × 1 file each + 1 sub-file in kurdish/Literature
        assert len(catalog.documents) == 5


def test_from_ktc_maps_categories():
    with tempfile.TemporaryDirectory() as tmpdir:
        ktc_root = _make_ktc_tree(tmpdir)
        catalog = CorpusCatalog.from_ktc(ktc_root)
        cats = {d.filename: d.category for d in catalog.documents}
        assert cats["economy/01s-ch01-2018.txt"] == "economics"
        assert cats["history/01s-ch01-2018.txt"] == "history"
        assert cats["physics/01s-ch01-2018.txt"] == "sciences"
        assert cats["kurdish/01s-ch01-2018.txt"] == "linguistics"
        assert cats["kurdish/Literature/07s-ch01-2015.txt"] == "linguistics"


def test_from_ktc_load_sentences():
    with tempfile.TemporaryDirectory() as tmpdir:
        ktc_root = _make_ktc_tree(tmpdir)
        catalog = CorpusCatalog.from_ktc(ktc_root)
        stats = catalog.load_sentences()
        # economy: 2, history: 3, physics: 1, kurdish: 2+1=3
        assert stats.total_sentences == 9
        assert stats.per_category["economics"] == 2
        assert stats.per_category["history"] == 3
        assert stats.per_category["sciences"] == 1
        assert stats.per_category["linguistics"] == 3


def test_from_ktc_balanced_sample():
    with tempfile.TemporaryDirectory() as tmpdir:
        ktc_root = _make_ktc_tree(tmpdir)
        catalog = CorpusCatalog.from_ktc(ktc_root)
        catalog.load_sentences()
        sampled = catalog.balanced_sample(target_sentences=6, min_per_category=1)
        total = sum(len(s) for s in sampled.values())
        assert total == 6


def test_from_ktc_missing_directory_handled():
    """from_ktc gracefully handles missing KTC category directories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Empty directory — no KTC subdirectories
        catalog = CorpusCatalog.from_ktc(tmpdir)
        assert len(catalog.documents) == 0
        stats = catalog.load_sentences()
        assert stats.total_sentences == 0


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
