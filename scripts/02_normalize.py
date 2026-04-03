"""
Step 2: Normalize and Clean Collected Text

Includes quality-gate filtering (PIPE-11):
  - Drops sentences with >30% non-Arabic-script characters
  - Drops sentences shorter than 5 tokens or longer than 150 tokens
  - Drops sentences matching known Wikipedia boilerplate patterns

Usage:
    python scripts/02_normalize.py [--input data/raw] [--output data/clean]
"""

import argparse
import logging
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.normalizer import SoraniNormalizer, deduplicate_sentences

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# PIPE-11: Quality-gate patterns
_ARABIC_SCRIPT_RE = re.compile(r'[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]')
_BOILERPLATE_PATTERNS = [
    "ئەم بابەتە بچووکە",       # Wikipedia stub notice
    "سەرچاوەکان دیاری نەکراون",  # "sources not specified"
    "ئەم بابەتە پێویستی بە",     # "this article needs"
    "بابەتی سەرەکی",             # "main article"
]
_MIN_TOKENS = 5
_MAX_TOKENS = 150
_MAX_NON_ARABIC_RATIO = 0.30


def passes_quality_gate(sentence: str) -> tuple[bool, str]:
    """Check whether a sentence passes quality filters.

    Returns:
        (passes, reason) — reason is empty string if passes is True.
    """
    tokens = sentence.split()
    if len(tokens) < _MIN_TOKENS:
        return False, "too_few_tokens"
    if len(tokens) > _MAX_TOKENS:
        return False, "too_many_tokens"

    # Check Arabic-script ratio (excluding whitespace and punctuation)
    non_space = re.sub(r'\s', '', sentence)
    if non_space:
        arabic_count = len(_ARABIC_SCRIPT_RE.findall(non_space))
        ratio = 1.0 - (arabic_count / len(non_space))
        if ratio > _MAX_NON_ARABIC_RATIO:
            return False, "low_arabic_ratio"

    # Check boilerplate
    for pattern in _BOILERPLATE_PATTERNS:
        if pattern in sentence:
            return False, "boilerplate"

    return True, ""


def main():
    parser = argparse.ArgumentParser(description="Normalize Sorani Kurdish text")
    parser.add_argument("--input", default="data/raw")
    parser.add_argument("--output", default="data/clean")
    parser.add_argument("--remove-diacritics", action="store_true")
    parser.add_argument("--skip-quality-gate", action="store_true",
                        help="Skip quality-gate filtering (PIPE-11)")
    args = parser.parse_args()
    
    normalizer = SoraniNormalizer(
        remove_diacritics=args.remove_diacritics,
    )
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    all_sentences = []
    dropped_short = 0
    quality_drops: dict[str, int] = {}
    
    for txt_file in input_path.glob("*.txt"):
        logger.info("Processing %s...", txt_file.name)
        with open(txt_file, "r", encoding="utf-8-sig", errors="replace") as f:
            for line in f:
                normalized = normalizer.normalize(line.strip())
                if normalized and len(normalized) > 20:
                    # PIPE-11: Quality-gate filtering
                    if not args.skip_quality_gate:
                        passes, reason = passes_quality_gate(normalized)
                        if not passes:
                            quality_drops[reason] = quality_drops.get(reason, 0) + 1
                            continue
                    all_sentences.append(normalized)
                elif normalized:
                    dropped_short += 1
    
    if dropped_short:
        logger.info("Dropped %d lines shorter than 20 chars", dropped_short)
    if quality_drops:
        for reason, count in quality_drops.items():
            logger.info("Quality gate — dropped %d lines (%s)", count, reason)
        logger.info("Quality gate total drops: %d", sum(quality_drops.values()))
    
    logger.info("Total sentences before dedup: %d", len(all_sentences))
    all_sentences = deduplicate_sentences(all_sentences)
    logger.info("Total sentences after dedup: %d", len(all_sentences))
    
    # Save cleaned corpus
    output_file = output_path / "clean_corpus.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        for sentence in all_sentences:
            f.write(sentence + "\n")
    
    logger.info("Clean corpus saved to %s", output_file)


if __name__ == "__main__":
    main()
