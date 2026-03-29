"""
Curriculum Learning Sampler

Progressively exposes harder training examples across epochs.
Early epochs train on easier (shorter) samples; later epochs include
the full dataset.

6B.8: Difficulty is measured by word count (not byte length), which
correlates more reliably with morphological complexity in Sorani Kurdish.
Byte length is misleading for Arabic-script text because multi-byte
UTF-8 characters inflate length without increasing linguistic complexity.
Optionally, agreement-edge density can augment the difficulty score.

PIPE-11: Supports pluggable difficulty metrics — word count (default)
or morphology-aware (agreement edge count + word count).
"""

import math
from typing import Iterator, Sequence, Optional

from torch.utils.data import Sampler


class CurriculumSampler(Sampler[int]):
    """Epoch-aware sampler that gradually increases difficulty.

    In epoch 0, only the easiest ``min_fraction`` of samples are used.
    By the final epoch, all samples are available.  Within each epoch's
    active subset the order is shuffled.

    Args:
        difficulties: per-sample difficulty scores (e.g. sentence length).
        total_epochs: total number of training epochs.
        min_fraction: fraction of data available in the first epoch.
        seed: base random seed for reproducibility.
    """

    def __init__(
        self,
        difficulties: Sequence[float],
        total_epochs: int = 30,
        min_fraction: float = 0.3,
        seed: int = 42,
    ):
        self.n = len(difficulties)
        self.total_epochs = max(1, total_epochs)
        self.min_fraction = max(0.1, min(1.0, min_fraction))
        self.seed = seed
        self._epoch = 0

        # Sort indices by difficulty (ascending = easiest first)
        self._sorted_indices = sorted(range(self.n), key=lambda i: difficulties[i])

    def set_epoch(self, epoch: int) -> None:
        """Set current epoch to adjust difficulty cutoff."""
        self._epoch = epoch

    def __len__(self) -> int:
        return self._active_count()

    def _active_count(self) -> int:
        """Number of samples available at the current epoch."""
        progress = min(self._epoch / max(1, self.total_epochs - 1), 1.0)
        fraction = self.min_fraction + (1.0 - self.min_fraction) * progress
        return max(1, math.ceil(fraction * self.n))

    def __iter__(self) -> Iterator[int]:
        import random
        active = self._active_count()
        indices = list(self._sorted_indices[:active])
        rng = random.Random(self.seed + self._epoch)
        rng.shuffle(indices)
        return iter(indices)


def compute_morphology_difficulty(
    sentences: Sequence[str],
    analyzer: Optional[object] = None,
    edge_weight: float = 0.5,
) -> list[float]:
    """Compute per-sentence difficulty scores using morphological complexity.

    Combines word count with agreement edge density from the agreement
    graph builder. Sentences with more agreement dependencies are scored
    as harder, even when short.

    Args:
        sentences: List of source sentences.
        analyzer: MorphologicalAnalyzer instance. If None, falls back
            to word count only.
        edge_weight: Weight given to edge count relative to word count.
            0.0 = pure word count; 1.0 = edges dominate.

    Returns:
        List of float difficulty scores (higher = harder).
    """
    if analyzer is None:
        return [float(len(s.split())) for s in sentences]

    from ..morphology.builder import build_agreement_graph

    scores = []
    for sent in sentences:
        n_words = len(sent.split())
        try:
            graph = build_agreement_graph(sent, analyzer)
            n_edges = len(graph.edges)
        except Exception:
            n_edges = 0
        score = n_words + edge_weight * n_edges
        scores.append(score)
    return scores
