"""
Tests for the baseline GEC model.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")


@pytest.fixture(scope="module")
def baseline_model():
    """Create a baseline GEC model (downloads byt5-small once)."""
    try:
        from src.model.baseline import BaselineGEC
        model = BaselineGEC(model_name="google/byt5-small", max_length=32)
        model.eval()
        return model
    except Exception:
        pytest.skip("byt5-small not available")


def test_baseline_init(baseline_model):
    """BaselineGEC initializes with correct attributes."""
    assert baseline_model.max_length == 32
    assert baseline_model.model_name == "google/byt5-small"
    print(f"  BaselineGEC initialized: model_name={baseline_model.model_name}")


def test_baseline_forward_returns_loss(baseline_model):
    """Forward pass with labels returns loss."""
    tok = baseline_model.tokenizer
    enc = tok("من دەچم", return_tensors="pt", max_length=32,
              truncation=True, padding="max_length")
    labels = enc["input_ids"].clone()
    with torch.no_grad():
        out = baseline_model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            labels=labels,
        )
    assert "loss" in out
    assert "logits" in out
    assert out["loss"].item() > 0
    print(f"  Forward loss: {out['loss'].item():.4f}")


def test_baseline_forward_requires_labels(baseline_model):
    """Forward pass without labels raises ValueError (T5 needs decoder input)."""
    tok = baseline_model.tokenizer
    enc = tok("من دەچم", return_tensors="pt", max_length=32,
              truncation=True, padding="max_length")
    import pytest as _pt
    with _pt.raises(ValueError):
        with torch.no_grad():
            baseline_model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
            )


def test_correct_returns_string(baseline_model):
    """correct() returns a non-empty string."""
    result = baseline_model.correct("من دەچم")
    assert isinstance(result, str)
    assert len(result) > 0
    print(f"  correct() output: '{result}'")


def test_correct_batch_returns_list(baseline_model):
    """correct_batch() returns list of strings."""
    results = baseline_model.correct_batch(["من دەچم", "تۆ دەزانیت"])
    assert isinstance(results, list)
    assert len(results) == 2
    for r in results:
        assert isinstance(r, str)
        assert len(r) > 0
    print(f"  correct_batch: {results}")


def test_correct_with_confidence_returns_tuple(baseline_model):
    """correct_with_confidence returns (str, float)."""
    text, confidence = baseline_model.correct_with_confidence("من دەچم")
    assert isinstance(text, str)
    assert isinstance(confidence, float)
    assert len(text) > 0
    print(f"  Confidence: {confidence:.4f} for '{text}'")


def test_correct_custom_beam_width(baseline_model):
    """correct() accepts custom beam width."""
    result = baseline_model.correct("سڵاو", num_beams=2)
    assert isinstance(result, str)


def test_correct_custom_length_penalty(baseline_model):
    """correct() accepts length_penalty parameter."""
    result = baseline_model.correct("من دەچم", length_penalty=0.8)
    assert isinstance(result, str)


if __name__ == "__main__":
    print("=== Baseline Model Tests ===")
    print("Run with: pytest tests/test_baseline_model.py -v")
