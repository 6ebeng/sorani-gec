"""
Tests for inter-rater agreement analysis module.
"""

import sys
import os
import json
import tempfile
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.evaluation.inter_rater import (
    load_ratings, percentage_agreement, cohens_kappa,
    compute_inter_rater_agreement,
)


def test_percentage_agreement_perfect():
    """100% agreement."""
    labels_a = ["good", "bad", "good"]
    labels_b = ["good", "bad", "good"]
    assert percentage_agreement(labels_a, labels_b) == 1.0


def test_percentage_agreement_none():
    """0% agreement."""
    labels_a = ["good", "good", "good"]
    labels_b = ["bad", "bad", "bad"]
    assert percentage_agreement(labels_a, labels_b) == 0.0


def test_percentage_agreement_partial():
    """Partial agreement: 2 out of 4."""
    labels_a = ["good", "bad", "good", "bad"]
    labels_b = ["good", "good", "bad", "bad"]
    assert percentage_agreement(labels_a, labels_b) == 0.5


def test_percentage_agreement_empty():
    """Empty labels → 0.0."""
    assert percentage_agreement([], []) == 0.0


def test_cohens_kappa_perfect():
    """Perfect agreement → kappa = 1.0."""
    labels_a = ["good", "bad", "good", "bad"]
    labels_b = ["good", "bad", "good", "bad"]
    k = cohens_kappa(labels_a, labels_b)
    assert k == 1.0, f"Expected 1.0, got {k}"


def test_cohens_kappa_empty():
    """Empty labels → kappa = 0."""
    assert cohens_kappa([], []) == 0.0


def test_cohens_kappa_chance():
    """Random-like agreement → low kappa."""
    # Construct labels where observed == expected by chance
    labels_a = ["good", "bad", "good", "bad"]
    labels_b = ["bad", "good", "bad", "good"]
    k = cohens_kappa(labels_a, labels_b)
    assert k <= 0.0, f"Opposite agreement should give kappa ≤ 0, got {k}"
    print(f"  Opposite agreement kappa: {k:.4f}")


def test_load_ratings_from_files(tmp_path):
    """load_ratings reads ratings_*.jsonl files."""
    # Create two rater files
    r1 = [
        {"source": "s1", "corrected": "c1", "rating": "good"},
        {"source": "s2", "corrected": "c2", "rating": "bad"},
    ]
    r2 = [
        {"source": "s1", "corrected": "c1", "rating": "good"},
        {"source": "s2", "corrected": "c2", "rating": "good"},
    ]
    (tmp_path / "ratings_alice.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in r1),
        encoding="utf-8",
    )
    (tmp_path / "ratings_bob.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in r2),
        encoding="utf-8",
    )

    ratings = load_ratings(tmp_path)
    assert "alice" in ratings
    assert "bob" in ratings
    assert len(ratings["alice"]) == 2
    assert len(ratings["bob"]) == 2
    print(f"  Loaded ratings for {list(ratings.keys())}")


def test_load_ratings_empty_dir(tmp_path):
    """load_ratings on empty dir returns empty dict."""
    ratings = load_ratings(tmp_path)
    assert ratings == {}


def test_compute_inter_rater_agreement(tmp_path):
    """Full pipeline: compute agreement between two raters."""
    r1 = [
        {"source": "s1", "corrected": "c1", "rating": "good"},
        {"source": "s2", "corrected": "c2", "rating": "bad"},
        {"source": "s3", "corrected": "c3", "rating": "good"},
    ]
    r2 = [
        {"source": "s1", "corrected": "c1", "rating": "good"},
        {"source": "s2", "corrected": "c2", "rating": "bad"},
        {"source": "s3", "corrected": "c3", "rating": "bad"},
    ]
    (tmp_path / "ratings_rater1.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in r1),
        encoding="utf-8",
    )
    (tmp_path / "ratings_rater2.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in r2),
        encoding="utf-8",
    )

    results = compute_inter_rater_agreement(tmp_path)
    assert len(results) >= 1
    for pair_key, metrics in results.items():
        assert "kappa" in metrics
        assert "agreement" in metrics
        assert "n_overlap" in metrics
        assert metrics["n_overlap"] == 3
        print(f"  {pair_key}: kappa={metrics['kappa']:.4f}, "
              f"agreement={metrics['agreement']:.2%}")


def test_compute_inter_rater_no_overlap(tmp_path):
    """Two raters with no overlapping items."""
    r1 = [{"source": "s1", "corrected": "c1", "rating": "good"}]
    r2 = [{"source": "s2", "corrected": "c2", "rating": "bad"}]
    (tmp_path / "ratings_x.jsonl").write_text(
        json.dumps(r1[0], ensure_ascii=False), encoding="utf-8",
    )
    (tmp_path / "ratings_y.jsonl").write_text(
        json.dumps(r2[0], ensure_ascii=False), encoding="utf-8",
    )

    results = compute_inter_rater_agreement(tmp_path)
    for pair_key, metrics in results.items():
        assert metrics["n_overlap"] == 0


if __name__ == "__main__":
    print("=== Inter-Rater Agreement Tests ===")
    test_percentage_agreement_perfect()
    test_percentage_agreement_none()
    test_percentage_agreement_partial()
    test_percentage_agreement_empty()
    test_cohens_kappa_perfect()
    test_cohens_kappa_empty()
    test_cohens_kappa_chance()
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        test_load_ratings_from_files(td)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        test_load_ratings_empty_dir(td)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        test_compute_inter_rater_agreement(td)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        test_compute_inter_rater_no_overlap(td)
    print("All inter-rater agreement tests passed!")
