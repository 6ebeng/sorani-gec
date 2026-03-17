"""
Step 3: Generate Synthetic Agreement Errors

Usage:
    python scripts/03_generate_errors.py [--input data/clean/clean_corpus.txt] [--output data/synthetic] [--target 50000]
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.errors.pipeline import ErrorPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic agreement errors")
    parser.add_argument("--input", default="data/clean/clean_corpus.txt")
    parser.add_argument("--output", default="data/synthetic")
    parser.add_argument("--target", type=int, default=50000, help="Target number of pairs")
    parser.add_argument("--error-rate", type=float, default=0.15)
    parser.add_argument("--corruption-ratio", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    pipeline = ErrorPipeline(
        error_rate=args.error_rate,
        seed=args.seed,
    )
    
    stats = pipeline.process_corpus(
        input_file=args.input,
        output_dir=args.output,
        target_pairs=args.target,
        corruption_ratio=args.corruption_ratio,
    )
    
    logger.info("Synthetic corpus generation complete.")
    logger.info("Stats: %s", stats)


if __name__ == "__main__":
    main()
