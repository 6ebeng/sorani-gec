"""
Tests for the ensemble GEC model.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")

if HAS_TORCH:
    from src.model.ensemble import EnsembleGEC


class MockModel(nn.Module):
    """Lightweight mock that returns a fixed correction."""

    def __init__(self, output: str):
        super().__init__()
        self._output = output
        # EnsembleGEC needs at least one parameter to iterate
        self._dummy = nn.Parameter(torch.zeros(1))

    def correct(self, text: str, num_beams: int = 4) -> str:
        return self._output


def test_ensemble_init():
    """EnsembleGEC initializes with models and strategy."""
    m1 = MockModel("a")
    m2 = MockModel("b")
    ensemble = EnsembleGEC(models=[m1, m2], strategy="majority_vote")
    assert len(ensemble.models) == 2
    assert ensemble.strategy == "majority_vote"


def test_ensemble_majority_vote_unanimous():
    """Majority vote with unanimous models returns the common answer."""
    m1 = MockModel("ڕاستکراو")
    m2 = MockModel("ڕاستکراو")
    m3 = MockModel("ڕاستکراو")
    ensemble = EnsembleGEC(models=[m1, m2, m3])
    result = ensemble.correct("تی")
    assert result == "ڕاستکراو"


def test_ensemble_majority_vote_split():
    """Majority vote picks the most common candidate."""
    m1 = MockModel("A")
    m2 = MockModel("B")
    m3 = MockModel("A")
    ensemble = EnsembleGEC(models=[m1, m2, m3])
    result = ensemble.correct("input")
    assert result == "A"


def test_ensemble_majority_vote_tie():
    """Tie-breaking: should return one of the tied candidates."""
    m1 = MockModel("X")
    m2 = MockModel("Y")
    ensemble = EnsembleGEC(models=[m1, m2])
    result = ensemble.correct("input")
    assert result in ("X", "Y")


def test_ensemble_correct_batch():
    """correct_batch returns a list with one result per input."""
    m1 = MockModel("corrected")
    m2 = MockModel("corrected")
    ensemble = EnsembleGEC(models=[m1, m2])
    results = ensemble.correct_batch(["a", "b", "c"])
    assert isinstance(results, list)
    assert len(results) == 3


def test_ensemble_single_model():
    """Ensemble with one model returns that model's output."""
    m1 = MockModel("سڵاو")
    ensemble = EnsembleGEC(models=[m1])
    assert ensemble.correct("x") == "سڵاو"


if __name__ == "__main__":
    print("=== Ensemble Model Tests ===")
    test_ensemble_init()
    test_ensemble_majority_vote_unanimous()
    test_ensemble_majority_vote_split()
    test_ensemble_majority_vote_tie()
    test_ensemble_correct_batch()
    test_ensemble_single_model()
    print("All ensemble tests passed!")
