"""
Step 1b: Sanitize Collected Text

Runs between collection (01_collect_data.py) and normalization (02_normalize.py).
Removes URLs, citation brackets, wiki templates, mojibake, non-prose lines,
and near-duplicates. Optionally re-applies Sorani language detection.

CRIT-2 in gap_analysis_and_missing_features.md.

Usage:
    python scripts/01b_sanitize.py [--input data/raw] [--output data/sanitized]
    python scripts/01b_sanitize.py --input data/raw --sorani-detect
"""

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.sanitizer import SoraniSanitizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Sanitize collected Sorani Kurdish text",
    )
    parser.add_argument("--input", default="data/raw",
                        help="Directory containing raw .txt files")
    parser.add_argument("--output", default="data/sanitized",
                        help="Directory to write sanitized files")
    parser.add_argument("--sorani-detect", action="store_true", default=False,
                        help="Re-apply Sorani language detection per sentence")
    parser.add_argument("--near-dup-threshold", type=float, default=0.90,
                        help="Jaccard threshold for near-duplicate detection")
    parser.add_argument("--min-tokens", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--preserve-categories", action="store_true", default=False,
                        help="If set, read tab-separated category\\tsentence lines "
                             "and preserve the category column through sanitization")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    detector = None
    if args.sorani_detect:
        from src.data.sorani_detector import SoraniDetector
        detector = SoraniDetector()
        logger.info("Sorani language detection enabled")

    sanitizer = SoraniSanitizer(
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        near_dup_threshold=args.near_dup_threshold,
        sorani_detector=detector,
    )

    total_in = 0
    total_out = 0

    txt_files = sorted(input_path.glob("*.txt"))
    if not txt_files:
        logger.error("No .txt files found in %s", input_path)
        raise SystemExit(1)

    all_sanitized: list[str] = []
    # When preserving categories, store (category, sentence) pairs
    all_categorized: list[tuple[str, str]] = []

    for txt_file in txt_files:
        logger.info("Reading %s...", txt_file.name)
        with open(txt_file, "r", encoding="utf-8-sig", errors="replace") as f:
            raw_lines = [line.strip() for line in f if line.strip()]
        total_in += len(raw_lines)

        if args.preserve_categories:
            # Split category\tsentence; lines without tab go to "general"
            categories: list[str] = []
            sentences: list[str] = []
            for raw in raw_lines:
                if "\t" in raw:
                    cat, sent = raw.split("\t", 1)
                    categories.append(cat)
                    sentences.append(sent)
                else:
                    categories.append("general")
                    sentences.append(raw)
            cleaned = sanitizer.sanitize_corpus(sentences)
            cleaned_set = set(cleaned)
            for cat, sent in zip(categories, sentences):
                if sent in cleaned_set:
                    all_categorized.append((cat, sent))
                    cleaned_set.discard(sent)
            total_out += len(all_categorized) - (total_out if total_out else 0)
        else:
            cleaned = sanitizer.sanitize_corpus(raw_lines)
            total_out += len(cleaned)
            all_sanitized.extend(cleaned)

        logger.info(
            "  %s: %d -> %d lines",
            txt_file.name, len(raw_lines),
            len(all_categorized) if args.preserve_categories else len(cleaned),
        )

    # Write combined output
    out_file = output_path / "sanitized_corpus.txt"
    with open(out_file, "w", encoding="utf-8") as f:
        if args.preserve_categories:
            for cat, sent in all_categorized:
                f.write("%s\t%s\n" % (cat, sent))
            total_out = len(all_categorized)
        else:
            for line in all_sanitized:
                f.write(line + "\n")

    logger.info("=== Sanitization Summary ===")
    logger.info("Total input lines:  %d", total_in)
    logger.info("Total output lines: %d", total_out)
    logger.info("Total dropped:      %d (%.1f%%)",
                total_in - total_out,
                100.0 * (total_in - total_out) / max(total_in, 1))
    if sanitizer.stats:
        logger.info("Drop reasons:")
        for reason, count in sorted(sanitizer.stats.items(), key=lambda x: -x[1]):
            logger.info("  %-25s %d", reason, count)
    logger.info("Sanitized corpus saved to %s", out_file)


if __name__ == "__main__":
    main()
