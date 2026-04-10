"""
Step 1c: Balance Corpus Across Academic Categories

Reads raw or sanitized corpus files, assigns each source document to an
academic discipline category, and samples evenly so that no single field
dominates the training data.

Runs after collection (01_collect_data.py) and optionally after
sanitization (01b_sanitize.py).

Usage:
    python scripts/01c_balance_corpus.py --input data/raw --output data/balanced/balanced_corpus.txt
    python scripts/01c_balance_corpus.py --input data/raw --catalog corpus_catalog.json --target 50000
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.corpus_catalog import CorpusCatalog

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Balance corpus across academic discipline categories",
    )
    parser.add_argument(
        "--input", required=True,
        help="Directory containing source .txt files (one per document)",
    )
    parser.add_argument(
        "--output", required=True,
        help="Path for the balanced corpus output file (tab-separated: category\\tsentence)",
    )
    parser.add_argument(
        "--catalog", default=None,
        help="Optional JSON catalog mapping filenames to categories. "
             "If omitted, categories are inferred from filenames.",
    )
    parser.add_argument(
        "--target", type=int, default=50000,
        help="Target number of sentences in the balanced corpus (default: 50000)",
    )
    parser.add_argument(
        "--min-per-category", type=int, default=100,
        help="Minimum sentences for a category to be included (default: 100)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible sampling (default: 42)",
    )
    parser.add_argument(
        "--save-catalog", default=None,
        help="If set, write the inferred/loaded catalog to this JSON path",
    )
    args = parser.parse_args()

    catalog = CorpusCatalog(
        corpus_dir=args.input,
        catalog_path=args.catalog,
        seed=args.seed,
    )

    stats = catalog.load_sentences()
    logger.info("Corpus loaded: %d sentences, %d documents", stats.total_sentences, stats.total_documents)

    for cat, count in sorted(stats.per_category.items(), key=lambda x: -x[1]):
        logger.info("  %-20s %6d sentences", cat, count)

    out_stats = catalog.save_balanced_corpus(
        output_path=args.output,
        target_sentences=args.target,
        min_per_category=args.min_per_category,
    )

    logger.info("=== Balanced Corpus Summary ===")
    logger.info("Total sentences: %d (target: %d)", out_stats.total_sentences, args.target)
    for cat, count in sorted(out_stats.per_category.items(), key=lambda x: -x[1]):
        pct = 100.0 * count / max(out_stats.total_sentences, 1)
        logger.info("  %-20s %6d  (%5.1f%%)", cat, count, pct)

    if args.save_catalog:
        catalog.save_catalog(args.save_catalog)


if __name__ == "__main__":
    main()
