"""
Step 1: Collect Sorani Kurdish Text Data

Usage:
    python scripts/01_collect_data.py [--source wikipedia|local] [--output data/raw]
"""

import argparse
import logging
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.collector import CorpusCollector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Collect Sorani Kurdish text data")
    parser.add_argument("--source", choices=["wikipedia", "local", "categorized", "ktc", "all"],
                        default="all")
    parser.add_argument("--output", default="data/raw")
    parser.add_argument("--local-dir", default=None, help="Path to local text files")
    parser.add_argument("--max-articles", type=int, default=5000)
    parser.add_argument("--catalog", default=None,
                        help="JSON catalog mapping source files to academic categories "
                             "(used with --source categorized)")
    parser.add_argument("--categorized-dir", default=None,
                        help="Directory with category subdirectories or flat files "
                             "plus a catalog JSON (used with --source categorized)")
    parser.add_argument("--ktc-dir", default="data/ktc",
                        help="Path to cloned KTC repo (used with --source ktc)")
    args = parser.parse_args()
    
    collector = CorpusCollector(output_dir=args.output)
    
    if args.source in ("wikipedia", "all"):
        logger.info("Collecting from Sorani Kurdish Wikipedia...")
        n = collector.collect_wikipedia()
        logger.info("Wikipedia: %d sentences collected", n)
    
    if args.source in ("local", "all") and args.local_dir:
        logger.info("Collecting from local files: %s", args.local_dir)
        n = collector.collect_from_text_files(args.local_dir, source_name="academic")
        logger.info("Local: %d sentences collected", n)

    if args.source == "categorized" and args.categorized_dir:
        logger.info("Collecting categorized corpus from: %s", args.categorized_dir)
        n = collector.collect_categorized(
            input_dir=args.categorized_dir,
            catalog_path=args.catalog,
        )
        logger.info("Categorized: %d sentences collected", n)

    if args.source in ("ktc", "all"):
        logger.info("Collecting from KTC (Kurdish Textbooks Corpus): %s", args.ktc_dir)
        n = collector.collect_from_ktc(ktc_dir=args.ktc_dir)
        logger.info("KTC: %d sentences collected", n)
    
    collector.save_stats()
    logger.info("Total: %d sentences", collector.stats['total_sentences'])
    logger.info("Data collection complete.")


if __name__ == "__main__":
    main()
