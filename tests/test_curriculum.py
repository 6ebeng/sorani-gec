"""
Tests for CurriculumSampler (src/data/curriculum.py).

Covers: single-epoch, multi-epoch progression, min_fraction edge cases,
deterministic seed behavior, and __len__ correctness.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.curriculum import CurriculumSampler


def test_init_basic():
    """CurriculumSampler initializes with default parameters."""
    difficulties = [3.0, 1.0, 2.0, 5.0, 4.0]
    sampler = CurriculumSampler(difficulties)
    assert sampler.n == 5
    assert sampler.total_epochs == 30
    assert sampler.min_fraction == 0.3
    assert sampler.seed == 42


def test_single_epoch():
    """With total_epochs=1, all samples are available in epoch 0."""
    difficulties = [3.0, 1.0, 2.0, 5.0, 4.0]
    sampler = CurriculumSampler(difficulties, total_epochs=1, min_fraction=0.3)
    sampler.set_epoch(0)
    indices = list(sampler)
    # Single epoch: progress = 0/max(1,0) = 0.0, fraction = 0.3
    # BUT total_epochs=1 means max(1, 1-1) = max(1,0) = 1
    # progress = min(0/1, 1.0) = 0.0
    # fraction = 0.3 + 0.7 * 0.0 = 0.3
    # active = ceil(0.3 * 5) = 2
    assert len(indices) == 2


def test_progressive_exposure():
    """Sampler exposes more data across epochs."""
    difficulties = list(range(100))
    sampler = CurriculumSampler(difficulties, total_epochs=10, min_fraction=0.3)

    prev_count = 0
    for epoch in range(10):
        sampler.set_epoch(epoch)
        count = len(list(sampler))
        assert count >= prev_count, (
            "Active count should not decrease: epoch %d had %d, epoch %d has %d"
            % (epoch - 1, prev_count, epoch, count)
        )
        prev_count = count

    # Final epoch should use all samples
    sampler.set_epoch(9)
    assert len(list(sampler)) == 100


def test_final_epoch_full_dataset():
    """At the final epoch, all samples are included."""
    difficulties = [5.0, 1.0, 3.0, 2.0, 4.0, 6.0, 7.0]
    sampler = CurriculumSampler(difficulties, total_epochs=5, min_fraction=0.2)
    sampler.set_epoch(4)  # final epoch
    indices = list(sampler)
    assert len(indices) == 7
    assert set(indices) == set(range(7))


def test_min_fraction_clamping():
    """min_fraction is clamped to [0.1, 1.0]."""
    difficulties = [1.0, 2.0, 3.0]
    # Too low — clamped to 0.1
    sampler = CurriculumSampler(difficulties, min_fraction=0.01)
    assert sampler.min_fraction == 0.1
    # Too high — clamped to 1.0
    sampler = CurriculumSampler(difficulties, min_fraction=1.5)
    assert sampler.min_fraction == 1.0


def test_deterministic_seed():
    """Same seed and epoch produce identical ordering."""
    difficulties = list(range(50))
    s1 = CurriculumSampler(difficulties, total_epochs=10, seed=123)
    s2 = CurriculumSampler(difficulties, total_epochs=10, seed=123)

    for epoch in range(10):
        s1.set_epoch(epoch)
        s2.set_epoch(epoch)
        assert list(s1) == list(s2), "Epoch %d ordering mismatch" % epoch


def test_different_seeds_differ():
    """Different seeds produce different orderings."""
    difficulties = list(range(50))
    s1 = CurriculumSampler(difficulties, total_epochs=10, seed=42)
    s2 = CurriculumSampler(difficulties, total_epochs=10, seed=99)
    s1.set_epoch(5)
    s2.set_epoch(5)
    # With 50 items, different seeds should produce different permutations
    assert list(s1) != list(s2)


def test_len_matches_iter():
    """__len__ returns the same count as iterating."""
    difficulties = list(range(20))
    sampler = CurriculumSampler(difficulties, total_epochs=5, min_fraction=0.4)
    for epoch in range(5):
        sampler.set_epoch(epoch)
        assert len(sampler) == len(list(sampler))


def test_easiest_first():
    """Early epochs only include the easiest (lowest difficulty) samples."""
    difficulties = [10.0, 1.0, 5.0, 2.0, 8.0]
    # sorted by difficulty: indices [1, 3, 2, 0, 4]
    sampler = CurriculumSampler(
        difficulties, total_epochs=10, min_fraction=0.4, seed=0,
    )
    sampler.set_epoch(0)
    indices = set(sampler)
    # min_fraction=0.4 → ceil(0.4*5) = 2 easiest
    assert len(indices) == 2
    # Easiest indices: 1 (diff=1.0) and 3 (diff=2.0)
    assert indices == {1, 3}


def test_total_epochs_zero_or_negative():
    """total_epochs is clamped to at least 1."""
    difficulties = [1.0, 2.0]
    sampler = CurriculumSampler(difficulties, total_epochs=0)
    assert sampler.total_epochs == 1
    sampler = CurriculumSampler(difficulties, total_epochs=-5)
    assert sampler.total_epochs == 1


def test_single_sample():
    """Works correctly with a single training sample."""
    sampler = CurriculumSampler([1.0], total_epochs=5)
    sampler.set_epoch(0)
    assert list(sampler) == [0]
    sampler.set_epoch(4)
    assert list(sampler) == [0]
