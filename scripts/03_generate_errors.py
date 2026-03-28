"""
Step 3: Generate Synthetic Agreement Errors

Usage:
    python scripts/03_generate_errors.py [--input data/clean/clean_corpus.txt] [--output data/synthetic] [--target 50000]
"""

import argparse
import hashlib
import logging
import os
import sys
from pathlib import Path

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

    if not Path(args.input).exists():
        logger.error("Input file not found: %s", args.input)
        raise SystemExit(1)

    if not (0.0 < args.error_rate <= 1.0):
        logger.error("--error-rate must be in (0, 1], got %s", args.error_rate)
        raise SystemExit(1)
    if not (0.0 <= args.corruption_ratio <= 1.0):
        logger.error("--corruption-ratio must be in [0, 1], got %s", args.corruption_ratio)
        raise SystemExit(1)

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

    # Fix 2.10: Structured error type distribution
    errors_by_type = stats.get("errors_by_type", {})
    total_errors = sum(errors_by_type.values()) if errors_by_type else 0
    if total_errors > 0:
        logger.info("=== Error Type Distribution (%d total errors) ===", total_errors)
        for etype, count in sorted(errors_by_type.items(), key=lambda x: -x[1]):
            pct = 100.0 * count / total_errors
            logger.info("  %-35s %6d  (%5.1f%%)", etype, count, pct)

    # Fix 2.14: Dataset provenance — hash of sorted source/target pairs
    src_path = Path(args.output) / "train.src"
    tgt_path = Path(args.output) / "train.tgt"
    if src_path.exists() and tgt_path.exists():
        h = hashlib.sha256()
        with open(src_path, "r", encoding="utf-8") as f:
            for line in sorted(f):
                h.update(line.encode("utf-8"))
        with open(tgt_path, "r", encoding="utf-8") as f:
            for line in sorted(f):
                h.update(line.encode("utf-8"))
        logger.info("Dataset SHA-256: %s", h.hexdigest())


if __name__ == "__main__":
    main()
