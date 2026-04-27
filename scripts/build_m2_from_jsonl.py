"""Build an M² gold file from a natural_test/sentences.jsonl dataset.

Given source/target pairs with human corrections, this uses the existing
word-level span aligner in `src/evaluation/m2_scorer.py` to emit an
ERRANT-compatible M² file usable by 07_evaluate.py's M² scorer.

Usage:
    python scripts/build_m2_from_jsonl.py \
        --input data/natural_test/sentences.jsonl \
        --output data/natural_test/annotations.m2
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.evaluation.m2_scorer import M2Edit, M2Sentence, write_m2_file  # noqa: E402
from src.evaluation.m2_scorer import _extract_span_edits  # noqa: E402
from src.data.tokenize import sorani_word_tokenize  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--default-type",
        default="M:OTHER",
        help="Error type used when `error_types` is empty in a record.",
    )
    args = parser.parse_args()

    sentences: list[M2Sentence] = []
    skipped = 0
    with args.input.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSON at line %d", line_num)
                skipped += 1
                continue

            src = (rec.get("source_text") or rec.get("source") or rec.get("corrupted") or "").strip()
            tgt = (rec.get("target_text") or rec.get("target") or rec.get("original") or "").strip()
            if not src or not tgt:
                skipped += 1
                continue
            # Skip trivial copy-pairs (FM4): no edit to align.
            if src == tgt:
                skipped += 1
                continue

            etypes = rec.get("error_types") or [
                e.get("type", args.default_type)
                for e in (rec.get("errors") or [])
                if isinstance(e, dict)
            ] or []
            type_str = etypes[0] if etypes else args.default_type

            src_words = sorani_word_tokenize(src)
            tgt_words = sorani_word_tokenize(tgt)
            spans = _extract_span_edits(src_words, tgt_words)

            s = M2Sentence(source=" ".join(src_words))
            for start, end, correction in spans:
                s.edits.append(M2Edit(
                    start=start,
                    end=end,
                    error_type=type_str,
                    correction=correction,
                    annotator=0,
                ))
            sentences.append(s)

    if not sentences:
        logger.error("No usable sentences. Did you fill target_text?")
        return 1

    write_m2_file(sentences, args.output)
    logger.info(
        "Wrote %d sentences (%d skipped) to %s",
        len(sentences), skipped, args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
