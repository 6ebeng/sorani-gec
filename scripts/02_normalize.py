"""
Step 2: Normalize and Clean Collected Text

Usage:
    python scripts/02_normalize.py [--input data/raw] [--output data/clean]
"""

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.normalizer import SoraniNormalizer, deduplicate_sentences

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Normalize Sorani Kurdish text")
    parser.add_argument("--input", default="data/raw")
    parser.add_argument("--output", default="data/clean")
    parser.add_argument("--remove-diacritics", action="store_true")
    args = parser.parse_args()
    
    normalizer = SoraniNormalizer(
        remove_diacritics=args.remove_diacritics,
    )
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    all_sentences = []
    dropped_short = 0
    
    for txt_file in input_path.glob("*.txt"):
        logger.info("Processing %s...", txt_file.name)
        with open(txt_file, "r", encoding="utf-8-sig", errors="replace") as f:
            for line in f:
                normalized = normalizer.normalize(line.strip())
                if normalized and len(normalized) > 20:
                    all_sentences.append(normalized)
                elif normalized:
                    dropped_short += 1
    
    if dropped_short:
        logger.info("Dropped %d lines shorter than 20 chars", dropped_short)
    
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
