"""
Error-type coverage regression tests for the released splits_v2 corpus.

These guard the Phase 2 data-integrity findings (audit rows 2.2/2.4):
  * the error labels in the data never drift away from the declared generators;
  * the documented coverage gaps stay documented (no silent regeneration);
  * the trivial/edited pair counts match the SHA-256 manifest the thesis numbers
    are reported on.

If splits_v2 is absent (fresh checkout without generated data) the data-backed
tests skip rather than fail.
"""

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.errors.pipeline import ErrorPipeline

ROOT = Path(__file__).resolve().parents[1]
SPLITS_V2 = ROOT / "data" / "splits_v2"


def declared_error_types() -> set[str]:
    """Canonical error labels, read straight from the generator registry."""
    return {g.error_type for g in ErrorPipeline().generators}


def _load_jsonl(path: Path) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _per_type_counts(records: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in records:
        for err in r.get("errors", []):
            et = err.get("type", err.get("error_type"))
            counts[et] = counts.get(et, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Code-level invariants (always run)
# ---------------------------------------------------------------------------

def test_declared_generator_count():
    """The pipeline registers exactly 25 generators with unique labels."""
    types = [g.error_type for g in ErrorPipeline().generators]
    assert len(types) == 25
    assert len(set(types)) == 25, "duplicate error_type labels in the registry"


# ---------------------------------------------------------------------------
# Data-backed coverage guards (skip if splits_v2 missing)
# ---------------------------------------------------------------------------

requires_data = pytest.mark.skipif(
    not (SPLITS_V2 / "train.jsonl").exists(),
    reason="splits_v2 not generated in this checkout",
)


@requires_data
def test_no_label_drift_against_generators():
    """Every error label present in the corpus is a declared generator label."""
    declared = declared_error_types()
    observed: set[str] = set()
    for name in ("train", "dev", "test"):
        observed |= set(_per_type_counts(_load_jsonl(SPLITS_V2 / f"{name}.jsonl")))
    unknown = observed - declared
    assert not unknown, f"corpus contains undeclared error labels: {sorted(unknown)}"


@requires_data
def test_documented_coverage_gap_polite_imperative():
    """polite_imperative fires nowhere in splits_v2 (documented dead generator).

    If this generator ever starts producing pairs the coverage section in
    Chapter 7 must be updated, so the test fails loudly on that change.
    """
    declared = declared_error_types()
    assert "polite_imperative" in declared
    total = 0
    for name in ("train", "dev", "test"):
        total += _per_type_counts(_load_jsonl(SPLITS_V2 / f"{name}.jsonl")).get(
            "polite_imperative", 0
        )
    assert total == 0


@requires_data
def test_trivial_edited_counts_match_manifest():
    """Per-split trivial/edited totals equal the SHA-256 manifest figures."""
    manifest = json.loads((SPLITS_V2 / "manifest.json").read_text(encoding="utf-8"))
    for name in ("train", "dev", "test"):
        records = _load_jsonl(SPLITS_V2 / f"{name}.jsonl")
        n_trivial = sum(1 for r in records if not r.get("errors", []))
        n_edited = len(records) - n_trivial
        assert n_trivial == manifest["splits"][name]["n_trivial"]
        assert n_edited == manifest["splits"][name]["n_edited"]


@requires_data
def test_corpus_covers_most_declared_types():
    """At least 24 of the 25 declared types appear somewhere in the corpus."""
    declared = declared_error_types()
    observed: set[str] = set()
    for name in ("train", "dev", "test"):
        observed |= set(_per_type_counts(_load_jsonl(SPLITS_V2 / f"{name}.jsonl")))
    covered = observed & declared
    assert len(covered) >= 24, (
        f"only {len(covered)}/25 declared types present; "
        f"absent: {sorted(declared - observed)}"
    )
