"""
Step 13: Curate a Natural Test Set for Evaluation

Reads raw Sorani Kurdish text (social media posts, student essays, forum text)
and produces a JSONL file of (source, target, error_type) triples for manual
annotation.  Two workflows are supported:

  1. **Seed mode** (default): reads a plain-text file of raw sentences (one per
     line) and writes a JSONL template where `target` is initially a copy of
     `source`.  A human annotator then corrects each `target` and fills in the
     `error_type` field.

  2. **Parallel mode**: reads a pre-annotated TSV file with columns
     <source>\\t<target>\\t<error_type> (tab-separated) and converts it to the
     project JSONL format directly.

The output JSONL is compatible with the splitter (04_split_data.py) and the
M² gold-file builder (14_build_m2_gold.py).

Usage:
    # Generate annotation template from raw sentences
    python scripts/13_curate_natural_testset.py --mode seed \\
        --input data/natural/raw_sentences.txt \\
        --output data/natural/testset.jsonl

    # Convert already-annotated TSV to JSONL
    python scripts/13_curate_natural_testset.py --mode parallel \\
        --input data/natural/annotated.tsv \\
        --output data/natural/testset.jsonl

    # Validate an existing annotation JSONL
    python scripts/13_curate_natural_testset.py --mode validate \\
        --input data/natural/testset.jsonl
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.normalizer import SoraniNormalizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Recognised error types — the 25 generators plus a catch-all for real errors
# that don't map neatly to a single synthetic category.
VALID_ERROR_TYPES = {
    "subject_verb", "noun_adjective", "clitic", "tense_agreement",
    "syntax_roles", "dialectal", "relative_clause", "adversative",
    "participle_swap", "orthography", "negative_concord",
    "vocative_imperative", "conditional_agreement", "adverb_verb_tense",
    "preposition_fusion", "demonstrative_contraction", "quantifier_agreement",
    "possessive_clitic", "polite_imperative", "spelling_confusion",
    "whitespace_error", "morpheme_order", "cross_clause_agreement",
    "word_order", "negative_verb_form",
    # Catch-alls for organic errors
    "lexical_choice", "punctuation", "spelling", "other",
}


def _normalise_line(line: str, normalizer: SoraniNormalizer) -> str:
    """Light normalisation: strip, normalise chars, but do NOT sentence-split."""
    text = line.strip()
    if not text:
        return ""
    # Character-level normalisation only (digits, heh, ZWNJ)
    text = normalizer.normalize_characters(text)
    return text


def seed_mode(input_path: Path, output_path: Path) -> int:
    """Read raw sentences and write a JSONL annotation template."""
    normalizer = SoraniNormalizer()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    seen: set[str] = set()
    with open(input_path, encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for lineno, raw_line in enumerate(fin, 1):
            text = _normalise_line(raw_line, normalizer)
            if not text:
                continue
            if text in seen:
                logger.debug("Skipping duplicate at line %d", lineno)
                continue
            seen.add(text)

            record = {
                "source": text,
                "target": text,  # annotator will correct this
                "error_type": "",  # annotator will fill this
                "annotated": False,
                "source_line": lineno,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    logger.info("Wrote %d annotation templates to %s", written, output_path)
    return written


def parallel_mode(input_path: Path, output_path: Path) -> int:
    """Read a TSV of <source>\\t<target>\\t<error_type> and write JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with open(input_path, encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for lineno, line in enumerate(fin, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                logger.warning("Line %d: expected ≥2 tab-separated fields, got %d",
                               lineno, len(parts))
                continue

            source = parts[0].strip()
            target = parts[1].strip()
            error_type = parts[2].strip() if len(parts) >= 3 else ""

            if not source:
                logger.warning("Line %d: empty source, skipping", lineno)
                continue

            record = {
                "source": source,
                "target": target,
                "error_type": error_type,
                "annotated": True,
                "source_line": lineno,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    logger.info("Wrote %d pairs to %s", written, output_path)
    return written


def validate_mode(input_path: Path) -> dict:
    """Validate an existing annotation JSONL and report statistics."""
    stats: dict = {
        "total": 0,
        "annotated": 0,
        "unannotated": 0,
        "has_correction": 0,
        "identity_pairs": 0,
        "missing_error_type": 0,
        "unknown_error_type": 0,
        "error_type_counts": {},
        "errors": [],
    }

    with open(input_path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                stats["errors"].append(f"Line {lineno}: invalid JSON — {exc}")
                continue

            stats["total"] += 1

            if "source" not in record or "target" not in record:
                stats["errors"].append(f"Line {lineno}: missing 'source' or 'target'")
                continue

            is_annotated = record.get("annotated", record["source"] != record["target"])
            if is_annotated:
                stats["annotated"] += 1
            else:
                stats["unannotated"] += 1

            if record["source"] != record["target"]:
                stats["has_correction"] += 1
            else:
                stats["identity_pairs"] += 1

            etype = record.get("error_type", "")
            if not etype:
                stats["missing_error_type"] += 1
            elif etype not in VALID_ERROR_TYPES:
                stats["unknown_error_type"] += 1
                stats["errors"].append(
                    f"Line {lineno}: unknown error_type '{etype}'"
                )
            else:
                stats["error_type_counts"][etype] = (
                    stats["error_type_counts"].get(etype, 0) + 1
                )

    # Print report
    print(f"\n{'='*50}")
    print(f"Natural Test Set Validation: {input_path}")
    print(f"{'='*50}")
    print(f"Total records:          {stats['total']}")
    print(f"Annotated:              {stats['annotated']}")
    print(f"Unannotated:            {stats['unannotated']}")
    print(f"With corrections:       {stats['has_correction']}")
    print(f"Identity (no error):    {stats['identity_pairs']}")
    print(f"Missing error_type:     {stats['missing_error_type']}")
    print(f"Unknown error_type:     {stats['unknown_error_type']}")

    if stats["error_type_counts"]:
        print(f"\nError type distribution:")
        for etype, count in sorted(stats["error_type_counts"].items(),
                                   key=lambda x: -x[1]):
            print(f"  {etype:30s} {count:5d}")

    if stats["errors"]:
        print(f"\nErrors ({len(stats['errors'])}):")
        for err in stats["errors"][:20]:
            print(f"  {err}")
        if len(stats["errors"]) > 20:
            print(f"  ... and {len(stats['errors']) - 20} more")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Curate a natural test set for Sorani Kurdish GEC evaluation"
    )
    parser.add_argument("--mode", choices=["seed", "parallel", "validate"],
                        default="seed",
                        help="seed: generate annotation template; "
                             "parallel: convert TSV to JSONL; "
                             "validate: check existing JSONL")
    parser.add_argument("--input", required=True,
                        help="Input file (txt for seed, tsv for parallel, jsonl for validate)")
    parser.add_argument("--output", default=None,
                        help="Output JSONL path (required for seed/parallel modes)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Input file does not exist: %s", input_path)
        sys.exit(1)

    if args.mode == "validate":
        validate_mode(input_path)
    else:
        if not args.output:
            logger.error("--output is required for %s mode", args.mode)
            sys.exit(1)
        output_path = Path(args.output)
        if args.mode == "seed":
            seed_mode(input_path, output_path)
        else:
            parallel_mode(input_path, output_path)


if __name__ == "__main__":
    main()
