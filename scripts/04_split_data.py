"""
Step 4: Split Synthetic Data into Train / Dev / Test

Usage:
    python scripts/04_split_data.py [--input data/synthetic/annotations.jsonl] [--output data/splits]
"""

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.splitter import run_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Split synthetic GEC data")
    parser.add_argument("--input", default="data/synthetic/annotations.jsonl")
    parser.add_argument("--output", default="data/splits")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stratify", default="error_type",
                        help="Key to stratify by (default: error_type)")
    parser.add_argument("--group-by", default=None,
                        help="Key to group by for article-level splitting (e.g. source_id)")
    parser.add_argument("--stratify-category", action="store_true", default=False,
                        help="Stratify splits so each has proportional category coverage "
                             "(requires 'category' field in input JSONL)")
    args = parser.parse_args()

    ratio_sum = args.train_ratio + args.dev_ratio + args.test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        logger.error("Split ratios must sum to 1.0, got %.6f", ratio_sum)
        raise SystemExit(1)

    if not Path(args.input).exists():
        logger.error("Input file not found: %s", args.input)
        raise SystemExit(1)

    stratify = args.stratify
    if args.stratify_category:
        stratify = "category"

    stats = run_split(
        input_path=Path(args.input),
        output_dir=Path(args.output),
        train_ratio=args.train_ratio,
        dev_ratio=args.dev_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        stratify_key=stratify,
        group_key=args.group_by,
    )

    logger.info("Split complete: %s", stats)


if __name__ == "__main__":
    main()
