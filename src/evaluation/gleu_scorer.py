"""
GLEU Scorer for Grammatical Error Correction

Computes the Generalized Language Evaluation Understanding (GLEU) metric
proposed by Napoles et al. (2015). GLEU modifies BLEU by penalising
n-grams that appear in the source but were left uncorrected when the
reference shows they should have been changed.

GLEU = min(precision, recall) over modified n-gram counts, averaged
across sentences and then across n-gram orders 1..4.

Reference:
    Napoles, C., Sakaguchi, K., Post, M., & Tetreault, J. (2015).
    Ground Truth for Grammatical Error Correction Metrics.
    ACL 2015.
"""

import logging
import math
from collections import Counter
from typing import Optional

logger = logging.getLogger(__name__)


def _ngrams(tokens: list[str], n: int) -> Counter:
    """Extract n-gram counts from a token list."""
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def _sentence_gleu(
    source_tokens: list[str],
    hypothesis_tokens: list[str],
    reference_tokens: list[str],
    max_order: int = 4,
) -> float:
    """Compute sentence-level GLEU.

    For each n-gram order, counts:
    - hyp_ngrams that appear in ref (rewarded)
    - hyp_ngrams that appear in src but NOT in ref (penalised — unchanged errors)

    Returns geometric mean of per-order scores, capped by a brevity penalty.
    """
    if not hypothesis_tokens or not reference_tokens:
        return 0.0  # caller logs the warning at corpus level

    log_gleu = 0.0
    effective_order = 0

    for n in range(1, max_order + 1):
        src_ng = _ngrams(source_tokens, n)
        hyp_ng = _ngrams(hypothesis_tokens, n)
        ref_ng = _ngrams(reference_tokens, n)

        if not hyp_ng:
            break

        # N-grams to reward: overlap between hyp and ref
        # N-grams to penalise: overlap between hyp and (src − ref),
        # i.e. source n-grams that should have been corrected but weren't
        src_only = src_ng - ref_ng  # n-grams in src that are NOT in ref

        tp = 0  # correct n-grams (in both hyp and ref)
        fp = 0  # unchanged error n-grams (in hyp and src_only)

        for ng, count in hyp_ng.items():
            ref_count = ref_ng.get(ng, 0)
            src_only_count = src_only.get(ng, 0)
            tp += min(count, ref_count)
            fp += min(count, src_only_count)

        total_hyp = sum(hyp_ng.values())
        total_ref = sum(ref_ng.values())

        # Precision-like: reward correct n-grams, penalise unchanged errors
        numerator = max(tp - fp, 0)
        # Smoothing: add 1 to avoid zero
        precision_n = (numerator + 1) / (total_hyp + 1)
        recall_n = (tp + 1) / (total_ref + 1)

        score_n = min(precision_n, recall_n)

        if score_n > 0:
            log_gleu += math.log(score_n)
            effective_order += 1

    if effective_order == 0:
        return 0.0

    # Brevity penalty (same as BLEU)
    bp = 1.0
    if len(hypothesis_tokens) < len(reference_tokens):
        bp = math.exp(1 - len(reference_tokens) / max(len(hypothesis_tokens), 1))

    return bp * math.exp(log_gleu / effective_order)


def compute_gleu(
    sources: list[str],
    hypotheses: list[str],
    references: list[str],
    max_order: int = 4,
    tokenize: Optional[type] = None,
) -> float:
    """Compute corpus-level GLEU score (averaged over sentences).

    Args:
        sources: Original corrupted sentences.
        hypotheses: Model-corrected sentences.
        references: Gold-standard clean sentences.
        max_order: Maximum n-gram order (default 4).
        tokenize: Callable to split string into tokens; defaults to str.split.

    Returns:
        GLEU score in [0, 1].
    """
    assert len(sources) == len(hypotheses) == len(references), \
        "All inputs must have the same length"

    _tok = tokenize or str.split
    total = 0.0

    for idx, (src, hyp, ref) in enumerate(zip(sources, hypotheses, references)):
        src_tokens = _tok(src)
        hyp_tokens = _tok(hyp)
        ref_tokens = _tok(ref)
        if not hyp_tokens or not ref_tokens:
            logger.warning("Empty hypothesis/reference at index %d", idx)
        total += _sentence_gleu(src_tokens, hyp_tokens, ref_tokens, max_order)

    return total / max(len(sources), 1)


def compute_gleu_per_sentence(
    sources: list[str],
    hypotheses: list[str],
    references: list[str],
    max_order: int = 4,
    tokenize: Optional[type] = None,
) -> list[float]:
    """Compute per-sentence GLEU scores.

    Args:
        sources: Original corrupted sentences.
        hypotheses: Model-corrected sentences.
        references: Gold-standard clean sentences.
        max_order: Maximum n-gram order (default 4).
        tokenize: Callable to split string into tokens; defaults to str.split.

    Returns:
        List of per-sentence GLEU scores.
    """
    assert len(sources) == len(hypotheses) == len(references)

    _tok = tokenize or str.split
    scores = []

    for src, hyp, ref in zip(sources, hypotheses, references):
        src_tokens = _tok(src)
        hyp_tokens = _tok(hyp)
        ref_tokens = _tok(ref)
        scores.append(_sentence_gleu(src_tokens, hyp_tokens, ref_tokens, max_order))

    return scores
