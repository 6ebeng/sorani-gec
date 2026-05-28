"""
Phase C R46 + R20: Measure morphological-analyser OOV rate on the GEC splits.

Reports per-split OOV rate (tokens not found in the 33,856-entry lexicon),
broken down by a rough POS category derived from positional heuristics.

Usage:
    cd Implementation/sorani-gec
    python scripts/measure_oov_rate.py [--splits-dir data/splits_v2] [--output results/oov_rate.json]
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_lexicon_tokens(lexicon_path: Path) -> set[str]:
    """Return the set of surface forms declared in the .dic file header section.

    Only reads the word entry lines (skipping the header count and affix flags).
    """
    tokens: set[str] = set()
    if not lexicon_path.exists():
        logger.error("Lexicon not found: %s", lexicon_path)
        return tokens
    with open(lexicon_path, encoding="utf-8", errors="replace") as f:
        first = True
        for line in f:
            line = line.strip()
            if not line:
                continue
            if first:
                first = False
                # First line is the count — skip it
                continue
            # Strip Hunspell flags (everything after / and the POS tags in <>)
            entry = line.split("/")[0].split("<")[0].strip()
            if entry:
                tokens.add(entry)
    logger.info("Loaded %d lexicon surface forms from %s", len(tokens), lexicon_path)
    return tokens


# Minimal Sorani tokenizer (whitespace + ZWNJ split)
_ZWNJ = "\u200c"

def tokenize(text: str) -> list[str]:
    """Tokenize by whitespace and ZWNJ boundaries, strip trailing punctuation."""
    text = text.replace(_ZWNJ, " ")
    words = text.split()
    clean = []
    for w in words:
        # Strip leading/trailing ASCII and Arabic punctuation
        w = re.sub(r'^[\.\،؟!؛:\(\)\[\]،]+|[\.\،؟!؛:\(\)\[\]،]+$', '', w)
        if w:
            clean.append(w)
    return clean


def is_likely_verb(token: str, tokens: list[str], idx: int) -> bool:
    """Very rough verb heuristic: token ends with common Sorani present-tense endings."""
    verb_suffixes = ("م", "ی", "ێت", "ین", "ن", "ێ")
    return any(token.endswith(s) for s in verb_suffixes)


def is_likely_proper_noun(token: str) -> bool:
    """Tokens that look like proper nouns (start with Arabic initial-form letters
    that are uncommon as common noun starters, or contain digits)."""
    # Heuristic: tokens that start with uppercase-equivalent or contain Latin
    return bool(re.search(r'[a-zA-Z0-9]', token))


def measure_oov(
    split_path: Path,
    lexicon_tokens: set[str],
    split_name: str,
) -> dict:
    """Measure OOV rate on the *target* (clean) side of a split."""
    records = []
    with open(split_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    total_tokens = 0
    oov_tokens   = 0
    oov_examples: list[str] = []

    cat_total: dict[str, int] = {"verb_like": 0, "proper_noun": 0, "other": 0}
    cat_oov:   dict[str, int] = {"verb_like": 0, "proper_noun": 0, "other": 0}

    for r in records:
        text = r.get("original", r.get("target", ""))
        tokens = tokenize(text)
        for idx, tok in enumerate(tokens):
            total_tokens += 1
            in_lex = tok in lexicon_tokens
            if not in_lex:
                oov_tokens += 1
                if len(oov_examples) < 20:
                    oov_examples.append(tok)

            # Rough POS category
            if is_likely_proper_noun(tok):
                cat = "proper_noun"
            elif is_likely_verb(tok, tokens, idx):
                cat = "verb_like"
            else:
                cat = "other"
            cat_total[cat] += 1
            if not in_lex:
                cat_oov[cat] += 1

    oov_rate = oov_tokens / total_tokens if total_tokens else 0.0

    result = {
        "split": split_name,
        "n_records": len(records),
        "total_tokens": total_tokens,
        "oov_tokens": oov_tokens,
        "oov_rate": round(oov_rate, 4),
        "oov_rate_pct": round(oov_rate * 100, 2),
        "per_pos_oov_rate": {
            cat: {
                "total": cat_total[cat],
                "oov": cat_oov[cat],
                "oov_rate_pct": round(
                    100.0 * cat_oov[cat] / cat_total[cat] if cat_total[cat] else 0, 2
                ),
            }
            for cat in cat_total
        },
        "oov_examples_first20": oov_examples,
    }

    logger.info(
        "%s: %d records, %d tokens, %d OOV (%.1f%%)",
        split_name, len(records), total_tokens, oov_tokens, oov_rate * 100,
    )
    for cat, vals in result["per_pos_oov_rate"].items():
        logger.info(
            "  %-15s  total=%6d  oov=%6d  (%.1f%%)",
            cat, vals["total"], vals["oov"], vals["oov_rate_pct"],
        )

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure OOV rate on GEC splits")
    parser.add_argument("--splits-dir", default="data/splits_v2")
    parser.add_argument("--lexicon",    default="data/hunspell/ckb-Arab.dic")
    parser.add_argument("--output",     default="results/oov_rate.json")
    args = parser.parse_args()

    lexicon_path = Path(args.lexicon)
    splits_dir   = Path(args.splits_dir)
    output_path  = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lexicon_tokens = load_lexicon_tokens(lexicon_path)
    if not lexicon_tokens:
        logger.error("Empty lexicon; check path: %s", lexicon_path)
        sys.exit(1)

    results = []
    for split_name in ("train", "dev", "test"):
        split_path = splits_dir / f"{split_name}.jsonl"
        if not split_path.exists():
            logger.warning("Split not found: %s — skipping", split_path)
            continue
        results.append(measure_oov(split_path, lexicon_tokens, split_name))

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("OOV results written to %s", output_path)


if __name__ == "__main__":
    main()
