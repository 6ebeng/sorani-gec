"""
F₀.₅ Scorer for Grammatical Error Correction

Computes Precision, Recall, and F₀.₅ score — the standard evaluation metric
for GEC tasks. F₀.₅ weights precision twice as much as recall, reflecting
that spurious corrections are worse than missed errors.

Provides two edit extraction modes:
- Word-level LCS (sentence_level_edits): simple diff returning (src, tgt) pairs.
  This mode conflates distinct morphological edits (a suffix change registers
  as a full-word substitution) and is NOT directly comparable with CoNLL-14
  or BEA-19 results that use ERRANT's span-level classification.
- Span-based (span_based_edits): position-tracking edits that classify changes
  as morphological (suffix/prefix change on same stem) vs full-word substitution,
  enabling closer comparison with ERRANT-style GEC benchmarks.

Deviation from ERRANT:
  ERRANT (Bryant et al., 2017) uses spaCy NLP models for automatic edit
  extraction and fine-grained error-type classification (M:NOUN:NUM, etc.).
  No spaCy model exists for Sorani Kurdish (ckb), so this module relies on
  character-overlap heuristics (≥50% shared prefix/suffix) instead of lemma
  matching. Results should be compared with ERRANT numbers only qualitatively.
  When reporting results, use evaluate_corpus_span() for the closest
  approximation to standard GEC evaluation.
"""

from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class GECMetrics:
    """GEC evaluation metrics."""
    precision: float
    recall: float
    f05: float
    tp: int       # true positives (correct corrections)
    fp: int       # false positives (spurious corrections)
    fn: int       # false negatives (missed errors)
    
    def __str__(self) -> str:
        return (
            f"P={self.precision:.4f}  R={self.recall:.4f}  F₀.₅={self.f05:.4f}  "
            f"(TP={self.tp}  FP={self.fp}  FN={self.fn})"
        )


@dataclass
class SpanEdit:
    """A span-based edit with position tracking and type classification."""
    src_start: int   # word index in source (inclusive)
    src_end: int     # word index in source (exclusive)
    src_text: str    # source text at this span
    tgt_text: str    # replacement text
    edit_type: str   # 'morphological', 'substitution', 'insertion', 'deletion'


def compute_f05(precision: float, recall: float) -> float:
    """Compute F₀.₅ from precision and recall."""
    beta = 0.5
    if precision + recall == 0:
        return 0.0
    return (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)


def sentence_level_edits(
    source: str,
    target: str,
    tokenize: Optional[Callable[[str], list[str]]] = None,
) -> list[tuple[str, str]]:
    """Extract word-level edits between source and target sentences.
    
    Returns list of (source_word, target_word) edit pairs.
    Simple word-level diff — for production use ERRANT or M2 format.
    """
    _tok = tokenize or str.split
    src_words = _tok(source)
    tgt_words = _tok(target)
    
    edits = []
    
    # Simple LCS-based diff
    m, n = len(src_words), len(tgt_words)
    
    # Build LCS table
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if src_words[i-1] == tgt_words[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    # Backtrack to find edits (including substitutions)
    i, j = m, n
    raw_edits: list[tuple[str, str]] = []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and src_words[i-1] == tgt_words[j-1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i-1][j-1] >= dp[i-1][j] and dp[i-1][j-1] >= dp[i][j-1]:
            raw_edits.append((src_words[i-1], tgt_words[j-1]))  # substitution
            i -= 1
            j -= 1
        elif j > 0 and (i == 0 or dp[i][j-1] >= dp[i-1][j]):
            raw_edits.append(("", tgt_words[j-1]))  # insertion
            j -= 1
        else:
            raw_edits.append((src_words[i-1], ""))  # deletion
            i -= 1
    
    return raw_edits


def evaluate_corpus(
    sources: list[str],     # corrupted input
    hypotheses: list[str],  # model output
    references: list[str],  # gold clean text
    tokenize: Optional[Callable[[str], list[str]]] = None,
) -> GECMetrics:
    """Evaluate GEC performance on a corpus.
    
    Uses simple word-level edit comparison.
    For rigorous evaluation, use ERRANT or M2Scorer.
    
    Args:
        sources: Original corrupted sentences
        hypotheses: Model-corrected sentences
        references: Gold-standard clean sentences
        tokenize: Optional tokenizer function; defaults to str.split
        
    Returns:
        GECMetrics with P, R, F₀.₅
    """
    assert len(sources) == len(hypotheses) == len(references), \
        "All inputs must have the same length"
    
    _tok = tokenize or str.split
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for src, hyp, ref in zip(sources, hypotheses, references):
        # Use LCS-based edit extraction to handle variable-length corrections
        # Use sorted lists (not sets) to preserve duplicate edits
        hyp_edits = sorted(sentence_level_edits(src, hyp, tokenize=_tok))
        ref_edits = sorted(sentence_level_edits(src, ref, tokenize=_tok))
        
        # Count TP/FP/FN using list matching to handle duplicates
        ref_remaining = list(ref_edits)
        tp = 0
        for edit in hyp_edits:
            if edit in ref_remaining:
                tp += 1
                ref_remaining.remove(edit)
        fp = len(hyp_edits) - tp
        fn = len(ref_remaining)
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f05 = compute_f05(precision, recall)
    
    return GECMetrics(
        precision=precision,
        recall=recall,
        f05=f05,
        tp=total_tp,
        fp=total_fp,
        fn=total_fn,
    )


def evaluate_corpus_by_type(
    sources: list[str],
    hypotheses: list[str],
    references: list[str],
    error_types: list[str],
    tokenize: Optional[Callable[[str], list[str]]] = None,
) -> dict[str, GECMetrics]:
    """Evaluate GEC performance per error type.

    Groups sentences by their error_type label and computes F0.5 for each.

    Args:
        sources: Original corrupted sentences.
        hypotheses: Model-corrected sentences.
        references: Gold-standard clean sentences.
        error_types: Error type label for each sentence.
        tokenize: Optional tokenizer function.

    Returns:
        Dict mapping error_type → GECMetrics.
    """
    assert len(sources) == len(hypotheses) == len(references) == len(error_types)

    _tok = tokenize or str.split
    buckets: dict[str, dict[str, int]] = {}

    for src, hyp, ref, etype in zip(sources, hypotheses, references, error_types):
        if etype not in buckets:
            buckets[etype] = {"tp": 0, "fp": 0, "fn": 0}

        hyp_edits = sorted(sentence_level_edits(src, hyp, tokenize=_tok))
        ref_edits = sorted(sentence_level_edits(src, ref, tokenize=_tok))

        ref_remaining = list(ref_edits)
        tp = 0
        for edit in hyp_edits:
            if edit in ref_remaining:
                tp += 1
                ref_remaining.remove(edit)
        buckets[etype]["tp"] += tp
        buckets[etype]["fp"] += len(hyp_edits) - tp
        buckets[etype]["fn"] += len(ref_remaining)

    results: dict[str, GECMetrics] = {}
    for etype, counts in sorted(buckets.items()):
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        results[etype] = GECMetrics(
            precision=p, recall=r, f05=compute_f05(p, r),
            tp=tp, fp=fp, fn=fn,
        )
    return results


# ============================================================================
# Span-based edit extraction (EVAL-1)
# ============================================================================

def _classify_edit(src_word: str, tgt_word: str) -> str:
    """Classify a single-word edit as morphological or substitution.

    Morphological: words share a stem (≥50% character overlap from either
    the start or end of the shorter word). This captures suffix/prefix
    changes typical of Sorani agreement errors.
    Substitution: entirely different lexemes.
    """
    if not src_word:
        return "insertion"
    if not tgt_word:
        return "deletion"

    # Character-level prefix overlap
    prefix = 0
    for a, b in zip(src_word, tgt_word):
        if a == b:
            prefix += 1
        else:
            break

    # Character-level suffix overlap
    suffix = 0
    for a, b in zip(reversed(src_word), reversed(tgt_word)):
        if a == b:
            suffix += 1
        else:
            break

    overlap = max(prefix, suffix)
    min_len = min(len(src_word), len(tgt_word))

    if min_len > 0 and overlap / min_len >= 0.5:
        return "morphological"
    return "substitution"


def span_based_edits(
    source: str,
    target: str,
    tokenize: Optional[Callable[[str], list[str]]] = None,
) -> list[SpanEdit]:
    """Extract span-based edits between source and target sentences.

    Unlike sentence_level_edits() which returns simple (src, tgt) pairs,
    this tracks word positions and classifies edits as morphological
    (suffix/prefix changes on a shared stem) vs full-word substitutions.
    Enables comparison with ERRANT-style span-based GEC evaluation.
    """
    _tok = tokenize or str.split
    src_words = _tok(source)
    tgt_words = _tok(target)

    m, n = len(src_words), len(tgt_words)

    # Build LCS table
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if src_words[i - 1] == tgt_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # Backtrack to collect edits with positional info
    i, j = m, n
    raw: list[SpanEdit] = []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and src_words[i - 1] == tgt_words[j - 1]:
            i -= 1
            j -= 1
        elif (i > 0 and j > 0
              and dp[i - 1][j - 1] >= dp[i - 1][j]
              and dp[i - 1][j - 1] >= dp[i][j - 1]):
            etype = _classify_edit(src_words[i - 1], tgt_words[j - 1])
            raw.append(SpanEdit(
                src_start=i - 1, src_end=i,
                src_text=src_words[i - 1], tgt_text=tgt_words[j - 1],
                edit_type=etype,
            ))
            i -= 1
            j -= 1
        elif j > 0 and (i == 0 or dp[i][j - 1] >= dp[i - 1][j]):
            raw.append(SpanEdit(
                src_start=i, src_end=i,
                src_text="", tgt_text=tgt_words[j - 1],
                edit_type="insertion",
            ))
            j -= 1
        else:
            raw.append(SpanEdit(
                src_start=i - 1, src_end=i,
                src_text=src_words[i - 1], tgt_text="",
                edit_type="deletion",
            ))
            i -= 1

    raw.reverse()

    # Merge adjacent edits of the same type into multi-word spans
    merged: list[SpanEdit] = []
    for edit in raw:
        if (merged
                and merged[-1].edit_type == edit.edit_type
                and merged[-1].src_end == edit.src_start):
            prev = merged[-1]
            src_text = f"{prev.src_text} {edit.src_text}".strip()
            tgt_text = f"{prev.tgt_text} {edit.tgt_text}".strip()
            merged[-1] = SpanEdit(
                src_start=prev.src_start, src_end=edit.src_end,
                src_text=src_text, tgt_text=tgt_text,
                edit_type=edit.edit_type,
            )
        else:
            merged.append(edit)

    return merged


def evaluate_corpus_span(
    sources: list[str],
    hypotheses: list[str],
    references: list[str],
    tokenize: Optional[Callable[[str], list[str]]] = None,
) -> tuple[GECMetrics, dict[str, GECMetrics]]:
    """Evaluate using span-based edits with per-edit-type breakdown.

    Position-aware edit matching: edits are compared by (src_start,
    src_end, tgt_text), so the same correction at a different position
    is not falsely counted as a true positive.

    Returns:
        Tuple of (overall GECMetrics, dict mapping edit_type → GECMetrics).
    """
    assert len(sources) == len(hypotheses) == len(references)

    _tok = tokenize or str.split
    total_tp = total_fp = total_fn = 0
    type_counts: dict[str, dict[str, int]] = {}

    for src, hyp, ref in zip(sources, hypotheses, references):
        hyp_edits = span_based_edits(src, hyp, tokenize=_tok)
        ref_edits = span_based_edits(src, ref, tokenize=_tok)

        # Position-aware keys for matching
        hyp_keys = [(e.src_start, e.src_end, e.tgt_text) for e in hyp_edits]
        ref_keys = list(range(len(ref_edits)))  # indices into ref_edits
        ref_key_tuples = [(e.src_start, e.src_end, e.tgt_text) for e in ref_edits]

        ref_remaining_idx = list(range(len(ref_edits)))
        ref_remaining_keys = list(ref_key_tuples)
        tp = 0
        for hk, he in zip(hyp_keys, hyp_edits):
            if hk in ref_remaining_keys:
                idx = ref_remaining_keys.index(hk)
                tp += 1
                ref_remaining_keys.pop(idx)
                ref_remaining_idx.pop(idx)
                etype = he.edit_type
                type_counts.setdefault(etype, {"tp": 0, "fp": 0, "fn": 0})
                type_counts[etype]["tp"] += 1
            else:
                etype = he.edit_type
                type_counts.setdefault(etype, {"tp": 0, "fp": 0, "fn": 0})
                type_counts[etype]["fp"] += 1

        # Count unmatched ref edits as FN
        for ri in ref_remaining_idx:
            etype = ref_edits[ri].edit_type
            type_counts.setdefault(etype, {"tp": 0, "fp": 0, "fn": 0})
            type_counts[etype]["fn"] += 1

        total_tp += tp
        total_fp += len(hyp_edits) - tp
        total_fn += len(ref_remaining_idx)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0

    overall = GECMetrics(
        precision=precision, recall=recall, f05=compute_f05(precision, recall),
        tp=total_tp, fp=total_fp, fn=total_fn,
    )

    per_type: dict[str, GECMetrics] = {}
    for etype, counts in sorted(type_counts.items()):
        etp, efp, efn = counts["tp"], counts["fp"], counts["fn"]
        ep = etp / (etp + efp) if (etp + efp) > 0 else 0.0
        er = etp / (etp + efn) if (etp + efn) > 0 else 0.0
        per_type[etype] = GECMetrics(
            precision=ep, recall=er, f05=compute_f05(ep, er),
            tp=etp, fp=efp, fn=efn,
        )

    return overall, per_type


# ============================================================================
# Sentence-level metrics (EVAL-3)
# ============================================================================

def evaluate_sentence(
    source: str,
    hypothesis: str,
    reference: str,
    tokenize: Optional[Callable[[str], list[str]]] = None,
) -> GECMetrics:
    """Compute F₀.₅ metrics for a single sentence.

    Enables per-sentence error analysis and identification of hardest
    sentence types (those with lowest sentence-level F₀.₅).
    """
    _tok = tokenize or str.split
    hyp_edits = sorted(sentence_level_edits(source, hypothesis, tokenize=_tok))
    ref_edits = sorted(sentence_level_edits(source, reference, tokenize=_tok))

    ref_remaining = list(ref_edits)
    tp = 0
    for edit in hyp_edits:
        if edit in ref_remaining:
            tp += 1
            ref_remaining.remove(edit)
    fp = len(hyp_edits) - tp
    fn = len(ref_remaining)

    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return GECMetrics(
        precision=p, recall=r, f05=compute_f05(p, r),
        tp=tp, fp=fp, fn=fn,
    )


def evaluate_corpus_with_sentences(
    sources: list[str],
    hypotheses: list[str],
    references: list[str],
    tokenize: Optional[Callable[[str], list[str]]] = None,
) -> tuple[GECMetrics, list[GECMetrics]]:
    """Evaluate GEC with both corpus-level and per-sentence metrics.

    The per-sentence list enables identification of hardest sentences
    (lowest F₀.₅) and distribution analysis across the test set.

    Returns:
        Tuple of (corpus_metrics, list of per-sentence GECMetrics).
    """
    assert len(sources) == len(hypotheses) == len(references)

    _tok = tokenize or str.split
    sentence_metrics: list[GECMetrics] = []
    total_tp = total_fp = total_fn = 0

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

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        sentence_metrics.append(GECMetrics(
            precision=p, recall=r, f05=compute_f05(p, r),
            tp=tp, fp=fp, fn=fn,
        ))

        total_tp += tp
        total_fp += fp
        total_fn += fn

    corpus_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    corpus_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0

    corpus_metrics = GECMetrics(
        precision=corpus_p, recall=corpus_r,
        f05=compute_f05(corpus_p, corpus_r),
        tp=total_tp, fp=total_fp, fn=total_fn,
    )

    return corpus_metrics, sentence_metrics


if __name__ == "__main__":
    # Quick test
    sources =    ["من دەچین بۆ قوتابخانە"]   # corrupted (wrong agreement)
    hypotheses = ["من دەچم بۆ قوتابخانە"]    # model correction
    references = ["من دەچم بۆ قوتابخانە"]    # gold
    
    metrics = evaluate_corpus(sources, hypotheses, references)
    print(metrics)
