"""
M2 Format Scorer for Grammatical Error Correction

Reads and writes the M2 annotation format used by the CoNLL shared tasks
and ERRANT. Computes span-level Precision, Recall, and F0.5.

M2 format:
    S <source sentence>
    A <start> <end>|||<error_type>|||<correction>|||REQUIRED|||-NONE-|||<annotator>
    (blank line between sentences)

ERRANT integration note:
    ERRANT (Bryant et al., 2017) is the de-facto standard GEC evaluation
    toolkit but depends on spaCy language models for automatic edit
    extraction and error-type classification. No spaCy model exists for
    Sorani Kurdish (ckb), making direct ERRANT use infeasible.

    This module provides M2-format-compatible parsing and scoring so that
    annotations produced by this pipeline can be evaluated against ERRANT
    output from other systems. The custom scorer here computes the same
    span-level P/R/F0.5 as ERRANT's ``errant_compare`` command but without
    the spaCy dependency. When a Sorani spaCy model becomes available,
    switching to ERRANT requires only replacing the edit extraction step;
    the M2 format and scoring logic remain identical.
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .f05_scorer import GECMetrics, compute_f05
from ..data.tokenize import sorani_word_tokenize

logger = logging.getLogger(__name__)


@dataclass
class M2Edit:
    """A single edit annotation in M2 format."""
    start: int
    end: int
    error_type: str
    correction: str
    annotator: int = 0


@dataclass
class M2Sentence:
    """A sentence with its M2 annotations."""
    source: str
    edits: list[M2Edit] = field(default_factory=list)


def parse_m2_file(path: Path) -> list[M2Sentence]:
    """Parse an M2-format file into structured sentence+edit objects."""
    sentences: list[M2Sentence] = []
    current: Optional[M2Sentence] = None

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("S "):
                current = M2Sentence(source=line[2:])
                sentences.append(current)
            elif line.startswith("A ") and current is not None:
                parts = line[2:].split("|||")
                if len(parts) >= 3:
                    span = parts[0].strip().split()
                    start = int(span[0])
                    end = int(span[1])
                    error_type = parts[1].strip()
                    correction = parts[2].strip()
                    annotator = int(parts[5]) if len(parts) > 5 and parts[5].strip().lstrip('-').isdigit() else 0
                    current.edits.append(M2Edit(
                        start=start, end=end,
                        error_type=error_type,
                        correction=correction,
                        annotator=annotator,
                    ))
    return sentences


def write_m2_file(sentences: list[M2Sentence], path: Path) -> None:
    """Write M2-format annotations to file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for sent in sentences:
            f.write(f"S {sent.source}\n")
            for edit in sent.edits:
                f.write(
                    f"A {edit.start} {edit.end}|||{edit.error_type}|||"
                    f"{edit.correction}|||REQUIRED|||-NONE-|||{edit.annotator}\n"
                )
            f.write("\n")


def edits_from_sentences(
    source: str, hypothesis: str, reference: str
) -> tuple[set[tuple], set[tuple]]:
    """Extract word-level span edits as (start, end, correction) tuples.

    Returns (hypothesis_edits, reference_edits) for one sentence pair.
    """
    # Tokenize using shared Sorani tokenizer for consistent word boundaries
    src_words = sorani_word_tokenize(source)
    hyp_words = sorani_word_tokenize(hypothesis)
    ref_words = sorani_word_tokenize(reference)

    hyp_edits = _extract_span_edits(src_words, hyp_words)
    ref_edits = _extract_span_edits(src_words, ref_words)
    return hyp_edits, ref_edits


def _extract_span_edits(
    src_words: list[str], tgt_words: list[str]
) -> set[tuple[int, int, str]]:
    """Use LCS alignment to get span-level edits as (start, end, correction).

    Detects single-word substitutions (including suffix-level changes like
    clitic attachment) by preferring diagonal moves in the DP backtrack
    when two words share a common prefix of at least half the shorter
    word's length.
    """
    m, n = len(src_words), len(tgt_words)

    # LCS table
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if src_words[i - 1] == tgt_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # Backtrack for alignment — prefer diagonal (substitution) over
    # insert/delete when words are morphological variants.
    alignment: list[tuple[Optional[int], Optional[int]]] = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and src_words[i - 1] == tgt_words[j - 1]:
            alignment.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and _shares_stem(src_words[i - 1], tgt_words[j - 1]):
            # Suffix-level change: treat as substitution (diagonal)
            alignment.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif j > 0 and (i == 0 or dp[i][j - 1] >= dp[i - 1][j]):
            alignment.append((None, j - 1))
            j -= 1
        else:
            alignment.append((i - 1, None))
            i -= 1
    alignment.reverse()

    edits: set[tuple[int, int, str]] = set()
    edit_start: Optional[int] = None
    edit_src_end: Optional[int] = None
    correction_tokens: list[str] = []

    for src_idx, tgt_idx in alignment:
        if src_idx is not None and tgt_idx is not None:
            if src_words[src_idx] == (tgt_words[tgt_idx] if tgt_idx is not None else ""):
                # Match — flush any pending edit
                if edit_start is not None:
                    end = edit_src_end + 1 if edit_src_end is not None else edit_start
                    edits.add((edit_start, end, " ".join(correction_tokens)))
                    edit_start = None
                    correction_tokens = []
            else:
                # Substitution — record as a single-span edit
                if edit_start is not None:
                    end = edit_src_end + 1 if edit_src_end is not None else edit_start
                    edits.add((edit_start, end, " ".join(correction_tokens)))
                    correction_tokens = []
                edit_start = src_idx
                edit_src_end = src_idx
                correction_tokens = [tgt_words[tgt_idx]]
                edits.add((edit_start, edit_src_end + 1, " ".join(correction_tokens)))
                edit_start = None
                correction_tokens = []
        else:
            # Mismatch — accumulate edit
            if src_idx is not None:
                if edit_start is None:
                    edit_start = src_idx
                edit_src_end = src_idx
            if tgt_idx is not None:
                if edit_start is None:
                    edit_start = (src_idx if src_idx is not None
                                  else (edit_src_end + 1 if edit_src_end is not None else 0))
                correction_tokens.append(tgt_words[tgt_idx])

    # Flush final edit
    if edit_start is not None:
        end = edit_src_end + 1 if edit_src_end is not None else edit_start
        edits.add((edit_start, end, " ".join(correction_tokens)))

    return edits


def _shares_stem(word_a: str, word_b: str) -> bool:
    """Check if two words share a common prefix ≥ 50% of the shorter word.

    This catches suffix-level changes like clitic attachment (کتێب → کتێبەکە)
    or agreement suffix swaps (دەچین → دەچم) without requiring a full
    morphological analyzer.
    """
    min_len = min(len(word_a), len(word_b))
    if min_len == 0:
        return False
    prefix = 0
    for a, b in zip(word_a, word_b):
        if a == b:
            prefix += 1
        else:
            break
    return prefix >= min_len * 0.5


def evaluate_m2(
    sources: list[str],
    hypotheses: list[str],
    references: list[str],
) -> GECMetrics:
    """Compute span-level F0.5 using M2-style edit extraction."""
    total_tp, total_fp, total_fn = 0, 0, 0

    for src, hyp, ref in zip(sources, hypotheses, references):
        hyp_edits, ref_edits = edits_from_sentences(src, hyp, ref)
        tp = len(hyp_edits & ref_edits)
        fp = len(hyp_edits - ref_edits)
        fn = len(ref_edits - hyp_edits)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f05 = compute_f05(precision, recall)

    return GECMetrics(
        precision=precision, recall=recall, f05=f05,
        tp=total_tp, fp=total_fp, fn=total_fn,
    )


def evaluate_m2_by_type(
    sources: list[str],
    hypotheses: list[str],
    m2_sentences: list[M2Sentence],
) -> dict[str, GECMetrics]:
    """Compute per-error-type F0.5 from M2 annotations.

    Groups reference edits by error_type and computes metrics per type.
    Unmatched hypothesis edits are counted as false positives under
    the nearest reference edit's type, or ``"spurious"`` if no reference
    edits exist for the sentence.
    """
    type_counts: dict[str, dict[str, int]] = {}

    for src, hyp, m2_sent in zip(sources, hypotheses, m2_sentences):
        hyp_edits = _extract_span_edits(sorani_word_tokenize(src), sorani_word_tokenize(hyp))
        matched_hyp: set[tuple[int, int, str]] = set()

        for m2_edit in m2_sent.edits:
            etype = m2_edit.error_type
            if etype not in type_counts:
                type_counts[etype] = {"tp": 0, "fp": 0, "fn": 0}
            ref_edit = (m2_edit.start, m2_edit.end, m2_edit.correction)
            if ref_edit in hyp_edits:
                type_counts[etype]["tp"] += 1
                matched_hyp.add(ref_edit)
            else:
                type_counts[etype]["fn"] += 1

        # Count unmatched hypothesis edits as false positives.
        # Attribute to nearest reference edit type by span distance;
        # fall back to "spurious" when no reference edits exist.
        unmatched = hyp_edits - matched_hyp
        if unmatched:
            ref_spans = [
                (e.start, e.end, e.error_type) for e in m2_sent.edits
            ]
            for h_start, h_end, _ in unmatched:
                fp_type = "spurious"
                best_dist = float("inf")
                for r_start, r_end, r_etype in ref_spans:
                    dist = abs(h_start - r_start) + abs(h_end - r_end)
                    if dist < best_dist:
                        best_dist = dist
                        fp_type = r_etype
                if fp_type not in type_counts:
                    type_counts[fp_type] = {"tp": 0, "fp": 0, "fn": 0}
                type_counts[fp_type]["fp"] += 1

    results: dict[str, GECMetrics] = {}
    for etype, counts in type_counts.items():
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        results[etype] = GECMetrics(
            precision=p, recall=r, f05=compute_f05(p, r),
            tp=tp, fp=fp, fn=fn,
        )
    return results
