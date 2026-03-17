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
    parser.add_argument("--source", choices=["wikipedia", "local", "all"], default="all")
    parser.add_argument("--output", default="data/raw")
    parser.add_argument("--local-dir", default=None, help="Path to local text files")
    parser.add_argument("--max-articles", type=int, default=5000)
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
    
    collector.save_stats()
    logger.info("Total: %d sentences", collector.stats['total_sentences'])
    logger.info("Data collection complete.")


if __name__ == "__main__":
    main()
