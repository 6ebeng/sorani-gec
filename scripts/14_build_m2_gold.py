"""
Step 14: Build Gold M² Files from Annotated JSONL

Reads an annotated JSONL file (produced by 13_curate_natural_testset.py or the
synthetic pipeline) and converts it to the CoNLL M² format used by standard GEC
evaluation tools.

For each pair, word-level edits are extracted via LCS alignment between the
source and target, then tagged with the error_type from the annotation.

Usage:
    # From annotated natural test set
    python scripts/14_build_m2_gold.py \\
        --input data/natural/testset.jsonl \\
        --output data/natural/gold.m2

    # From synthetic splits (e.g. test split)
    python scripts/14_build_m2_gold.py \\
        --input data/splits/test.jsonl \\
        --output data/splits/test.gold.m2

    # With annotator ID (for multi-annotator setups)
    python scripts/14_build_m2_gold.py \\
        --input data/natural/testset.jsonl \\
        --output data/natural/gold.m2 \\
        --annotator 0
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.tokenize import sorani_word_tokenize
from src.evaluation.m2_scorer import M2Edit, M2Sentence, write_m2_file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _extract_edits(
    src_words: list[str],
    tgt_words: list[str],
    error_type: str,
    annotator: int,
) -> list[M2Edit]:
    """Extract word-level edits between source and target using LCS alignment."""
    m, n = len(src_words), len(tgt_words)

    # LCS DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if src_words[i - 1] == tgt_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # Backtrack — prefer diagonal (substitution) when words share a prefix
    alignment: list[tuple[int | None, int | None]] = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and src_words[i - 1] == tgt_words[j - 1]:
            alignment.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif i > 0 and j > 0:
            sw, tw = src_words[i - 1], tgt_words[j - 1]
            min_len = min(len(sw), len(tw))
            common_prefix = sum(1 for a, b in zip(sw, tw) if a == b)
            if common_prefix >= min_len // 2 and min_len > 1:
                alignment.append((i - 1, j - 1))
                i -= 1
                j -= 1
            elif dp[i - 1][j] >= dp[i][j - 1]:
                alignment.append((i - 1, None))
                i -= 1
            else:
                alignment.append((None, j - 1))
                j -= 1
        elif i > 0:
            alignment.append((i - 1, None))
            i -= 1
        else:
            alignment.append((None, j - 1))
            j -= 1

    alignment.reverse()

    # Extract edits from alignment
    edits: list[M2Edit] = []
    idx = 0
    while idx < len(alignment):
        si, ti = alignment[idx]

        if si is not None and ti is not None:
            # Match or substitution
            if src_words[si] != tgt_words[ti]:
                edits.append(M2Edit(
                    start=si, end=si + 1,
                    error_type=error_type or "UNK",
                    correction=tgt_words[ti],
                    annotator=annotator,
                ))
            idx += 1
        elif si is not None and ti is None:
            # Deletion in target (source word was extra = insertion error)
            del_start = si
            while idx < len(alignment) and alignment[idx] == (si, None):
                si = alignment[idx][0]
                idx += 1
                if idx < len(alignment) and alignment[idx][0] is not None and alignment[idx][1] is None:
                    si = alignment[idx][0]
                else:
                    break
            edits.append(M2Edit(
                start=del_start, end=si + 1,
                error_type=error_type or "UNK",
                correction="",
                annotator=annotator,
            ))
        elif si is None and ti is not None:
            # Insertion in target (missing from source)
            ins_words = [tgt_words[ti]]
            ins_pos = 0
            # Find insertion position in source
            for prev_idx in range(idx - 1, -1, -1):
                if alignment[prev_idx][0] is not None:
                    ins_pos = alignment[prev_idx][0] + 1
                    break
            idx += 1
            while idx < len(alignment) and alignment[idx][0] is None:
                ins_words.append(tgt_words[alignment[idx][1]])
                idx += 1
            edits.append(M2Edit(
                start=ins_pos, end=ins_pos,
                error_type=error_type or "UNK",
                correction=" ".join(ins_words),
                annotator=annotator,
            ))
        else:
            idx += 1

    return edits


def build_m2(input_path: Path, output_path: Path, annotator: int = 0) -> int:
    """Convert annotated JSONL to M² gold file."""
    sentences: list[M2Sentence] = []
    skipped = 0

    with open(input_path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Line %d: invalid JSON, skipping", lineno)
                skipped += 1
                continue

            source = record.get("source", "")
            target = record.get("target", "")
            error_type = record.get("error_type", "")

            if not source:
                logger.warning("Line %d: empty source, skipping", lineno)
                skipped += 1
                continue

            src_words = sorani_word_tokenize(source)

            if source == target:
                # No correction needed — noop edit
                m2_sent = M2Sentence(source=source, edits=[
                    M2Edit(start=-1, end=-1, error_type="noop",
                           correction="-NONE-", annotator=annotator)
                ])
            else:
                tgt_words = sorani_word_tokenize(target)
                edits = _extract_edits(src_words, tgt_words, error_type, annotator)
                if not edits:
                    # Tokenization normalised away the difference
                    edits = [M2Edit(start=-1, end=-1, error_type="noop",
                                    correction="-NONE-", annotator=annotator)]
                m2_sent = M2Sentence(source=source, edits=edits)

            sentences.append(m2_sent)

    write_m2_file(sentences, output_path)
    logger.info(
        "Wrote %d sentences (%d edits) to %s [%d skipped]",
        len(sentences),
        sum(len(s.edits) for s in sentences),
        output_path,
        skipped,
    )
    return len(sentences)


def main():
    parser = argparse.ArgumentParser(
        description="Build gold M² files from annotated JSONL"
    )
    parser.add_argument("--input", required=True,
                        help="Annotated JSONL file (source/target/error_type)")
    parser.add_argument("--output", required=True,
                        help="Output M² file path")
    parser.add_argument("--annotator", type=int, default=0,
                        help="Annotator ID for M² annotations (default: 0)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Input file does not exist: %s", input_path)
        sys.exit(1)

    build_m2(input_path, Path(args.output), args.annotator)


if __name__ == "__main__":
    main()
