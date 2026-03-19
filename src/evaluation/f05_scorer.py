"""
F₀.₅ Scorer for Grammatical Error Correction

Computes Precision, Recall, and F₀.₅ score — the standard evaluation metric
for GEC tasks. F₀.₅ weights precision twice as much as recall, reflecting
that spurious corrections are worse than missed errors.
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
    
    # Backtrack to find edits
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and src_words[i-1] == tgt_words[j-1]:
            i -= 1
            j -= 1
        elif j > 0 and (i == 0 or dp[i][j-1] >= dp[i-1][j]):
            edits.append(("", tgt_words[j-1]))  # insertion
            j -= 1
        else:
            edits.append((src_words[i-1], ""))  # deletion
            i -= 1
    
    return edits


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
        hyp_edits = set(sentence_level_edits(src, hyp, tokenize=_tok))
        ref_edits = set(sentence_level_edits(src, ref, tokenize=_tok))
        
        # TP: edits in both hyp and ref
        tp = len(hyp_edits & ref_edits)
        # FP: edits in hyp but not in ref
        fp = len(hyp_edits - ref_edits)
        # FN: edits in ref but not in hyp
        fn = len(ref_edits - hyp_edits)
        
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


if __name__ == "__main__":
    # Quick test
    sources =    ["من دەچین بۆ قوتابخانە"]   # corrupted (wrong agreement)
    hypotheses = ["من دەچم بۆ قوتابخانە"]    # model correction
    references = ["من دەچم بۆ قوتابخانە"]    # gold
    
    metrics = evaluate_corpus(sources, hypotheses, references)
    print(metrics)
