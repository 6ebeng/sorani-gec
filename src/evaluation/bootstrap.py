"""Paired bootstrap significance testing for GEC system comparison.

Reviewer item R9: the headline baseline-vs-morphaware F0.5 gap, and any
baseline-vs-baseline gap, must come with a significance test rather than
two point estimates whose confidence intervals overlap. This module
implements a sentence-level paired bootstrap.

The unit of resampling is the sentence. For each of ``n_resamples`` draws we
sample sentence indices with replacement, recompute the corpus-level F0.5 for
both systems from the resampled per-sentence (tp, fp, fn) counts, and record
the difference ``delta = f05_a - f05_b``. The two-sided p-value is the fraction
of resamples whose delta sign disagrees with the observed delta (doubled, then
clamped to 1.0); the 95% interval is the empirical 2.5/97.5 percentile of the
resampled deltas.

F0.5 is a corpus-level ratio of summed counts, so it cannot be averaged across
sentences. Resampling the raw counts and recomputing the ratio inside each draw
keeps the statistic correct under the bootstrap.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Optional

from .f05_scorer import compute_f05, sentence_level_edits


@dataclass
class SentenceCounts:
    """Per-sentence edit-overlap counts for one system against the reference."""

    tp: int
    fp: int
    fn: int


@dataclass
class BootstrapResult:
    """Outcome of a paired bootstrap comparison of two systems."""

    f05_a: float
    f05_b: float
    delta: float          # f05_a - f05_b on the full corpus
    ci_low: float         # 2.5th percentile of resampled delta
    ci_high: float        # 97.5th percentile of resampled delta
    p_value: float        # two-sided
    n_resamples: int
    n_sentences: int

    def __str__(self) -> str:
        return (
            f"F0.5(A)={self.f05_a:.4f}  F0.5(B)={self.f05_b:.4f}  "
            f"delta={self.delta:+.4f}  95% CI [{self.ci_low:+.4f}, {self.ci_high:+.4f}]  "
            f"p={self.p_value:.4f}  (n={self.n_sentences}, B={self.n_resamples})"
        )


def _per_sentence_counts(
    sources: list[str],
    hypotheses: list[str],
    references: list[str],
    tokenize: Optional[Callable[[str], list[str]]] = None,
) -> list[SentenceCounts]:
    """Compute (tp, fp, fn) edit-overlap counts for every sentence.

    Mirrors the matching logic in ``evaluate_corpus`` exactly so that summing
    these counts reproduces the corpus F0.5.
    """
    _tok = tokenize or str.split
    out: list[SentenceCounts] = []
    for src, hyp, ref in zip(sources, hypotheses, references):
        hyp_edits = sorted(sentence_level_edits(src, hyp, tokenize=_tok))
        ref_edits = sorted(sentence_level_edits(src, ref, tokenize=_tok))
        ref_remaining = list(ref_edits)
        tp = 0
        for edit in hyp_edits:
            if edit in ref_remaining:
                tp += 1
                ref_remaining.remove(edit)
        fp = len(hyp_edits) - tp
        fn = len(ref_remaining)
        out.append(SentenceCounts(tp=tp, fp=fp, fn=fn))
    return out


def _corpus_f05(counts: list[SentenceCounts], indices: list[int]) -> float:
    tp = fp = fn = 0
    for i in indices:
        c = counts[i]
        tp += c.tp
        fp += c.fp
        fn += c.fn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return compute_f05(precision, recall)


def paired_bootstrap_f05(
    sources: list[str],
    hypotheses_a: list[str],
    hypotheses_b: list[str],
    references: list[str],
    n_resamples: int = 1000,
    seed: int = 42,
    tokenize: Optional[Callable[[str], list[str]]] = None,
) -> BootstrapResult:
    """Paired sentence-level bootstrap comparing system A against system B.

    Args:
        sources: corrupted input sentences.
        hypotheses_a: system A corrections.
        hypotheses_b: system B corrections.
        references: gold clean sentences.
        n_resamples: number of bootstrap draws.
        seed: RNG seed for reproducibility.
        tokenize: optional tokenizer; defaults to ``str.split``.

    Returns:
        BootstrapResult with point estimates, the 95% CI of the delta, and a
        two-sided p-value for the null hypothesis delta == 0.
    """
    n = len(sources)
    assert len(hypotheses_a) == len(hypotheses_b) == len(references) == n, \
        "All inputs must have the same length"

    counts_a = _per_sentence_counts(sources, hypotheses_a, references, tokenize)
    counts_b = _per_sentence_counts(sources, hypotheses_b, references, tokenize)

    all_idx = list(range(n))
    f05_a = _corpus_f05(counts_a, all_idx)
    f05_b = _corpus_f05(counts_b, all_idx)
    observed_delta = f05_a - f05_b

    rng = random.Random(seed)
    deltas: list[float] = []
    n_opposite = 0
    for _ in range(n_resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        d = _corpus_f05(counts_a, idx) - _corpus_f05(counts_b, idx)
        deltas.append(d)
        # Count resamples on the opposite side of zero from the observed delta.
        if observed_delta >= 0:
            if d <= 0:
                n_opposite += 1
        else:
            if d >= 0:
                n_opposite += 1

    deltas.sort()
    lo = deltas[int(0.025 * n_resamples)]
    hi = deltas[min(int(0.975 * n_resamples), n_resamples - 1)]
    p_value = min(1.0, 2.0 * n_opposite / n_resamples)

    return BootstrapResult(
        f05_a=f05_a,
        f05_b=f05_b,
        delta=observed_delta,
        ci_low=lo,
        ci_high=hi,
        p_value=p_value,
        n_resamples=n_resamples,
        n_sentences=n,
    )
