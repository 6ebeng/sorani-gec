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
    parser.add_argument("--spell-check", action="store_true", default=False,
                        help="Post-filter: discard pairs where corrupted side "
                             "has real misspellings (not intentional errors)")
    parser.add_argument("--validate-errors", action="store_true", default=False,
                        help="CRIT-6: reject error pairs where injected error "
                             "tokens are valid dictionary words")
    parser.add_argument("--preserve-categories", action="store_true", default=False,
                        help="Input has tab-separated category\\tsentence format; "
                             "category field is carried into annotations.jsonl")
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

    # If categories are preserved, strip the category column, process the
    # plain sentences, and then re-attach categories to the output JSONL.
    category_map: dict[int, str] = {}
    if args.preserve_categories:
        input_path = Path(args.input)
        tmp_input = input_path.with_name(input_path.stem + "_stripped.txt")
        with open(input_path, "r", encoding="utf-8") as fin, \
             open(tmp_input, "w", encoding="utf-8") as fout:
            idx = 0
            for line in fin:
                stripped = line.strip()
                if not stripped:
                    continue
                if "\t" in stripped:
                    cat, sent = stripped.split("\t", 1)
                    category_map[idx] = cat
                    fout.write(sent + "\n")
                else:
                    category_map[idx] = "general"
                    fout.write(stripped + "\n")
                idx += 1
        logger.info("Stripped categories from %d lines", len(category_map))
        args.input = str(tmp_input)
    
    try:
        stats = pipeline.process_corpus(
            input_file=args.input,
            output_dir=args.output,
            target_pairs=args.target,
            corruption_ratio=args.corruption_ratio,
            validate_errors=args.validate_errors,
        )
    finally:
        # PIPE-20: ensure temp file is cleaned up even if process_corpus raises
        if args.preserve_categories:
            tmp_path = Path(args.input)
            if tmp_path.name.endswith("_stripped.txt") and tmp_path.exists():
                tmp_path.unlink()
                logger.info("Cleaned up temp file: %s", tmp_path)

    logger.info("Synthetic corpus generation complete.")
    logger.info("Stats: %s", stats)

    # Inject category field into annotations.jsonl if categories were preserved
    if args.preserve_categories and category_map:
        import json as _json
        annotations_path = Path(args.output) / "annotations.jsonl"
        if annotations_path.exists():
            with open(annotations_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            with open(annotations_path, "w", encoding="utf-8") as f:
                for i, line in enumerate(lines):
                    record = _json.loads(line)
                    record["category"] = category_map.get(i, "general")
                    f.write(_json.dumps(record, ensure_ascii=False) + "\n")
            logger.info("Injected category field into %d annotations", len(lines))

    # PIPE-3: Optional spell-check post-filter
    if args.spell_check:
        from src.data.spell_checker import SoraniSpellChecker
        checker = SoraniSpellChecker()
        if checker.is_available():
            src_path = Path(args.output) / "train.src"
            tgt_path = Path(args.output) / "train.tgt"
            if src_path.exists() and tgt_path.exists():
                with open(src_path, "r", encoding="utf-8") as f:
                    src_lines = f.readlines()
                with open(tgt_path, "r", encoding="utf-8") as f:
                    tgt_lines = f.readlines()
                if len(src_lines) != len(tgt_lines):
                    logger.warning(
                        "src/tgt line count mismatch: %d vs %d",
                        len(src_lines), len(tgt_lines),
                    )
                kept_src, kept_tgt = [], []
                dropped = 0
                for s, t in zip(src_lines, tgt_lines):
                    # The target (clean) side should have no misspellings.
                    # Drop pairs where the clean side has misspelled words.
                    words = t.strip().split()
                    bad = sum(1 for w in words if not checker.is_correct(w))
                    if bad > len(words) * 0.3:  # >30% misspelled → suspect
                        dropped += 1
                        continue
                    kept_src.append(s)
                    kept_tgt.append(t)
                with open(src_path, "w", encoding="utf-8") as f:
                    f.writelines(kept_src)
                with open(tgt_path, "w", encoding="utf-8") as f:
                    f.writelines(kept_tgt)
                logger.info(
                    "Spell-check filter: kept %d, dropped %d pairs",
                    len(kept_src), dropped,
                )
        else:
            logger.warning("Spell checker not available — skipping post-filter")

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
