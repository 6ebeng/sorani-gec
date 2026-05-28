"""
Targeted test-set augmentation for splits_v2 (R10).

For error types with fewer than MIN_N examples in data/splits_v2/test.jsonl,
this script:
  1. Uses the original test-split sentences (data/splits/test.jsonl -> 'original'
     field) as the source pool to preserve split boundary integrity.
  2. Runs each target generator in isolation (single-edit only) on those sentences.
  3. Applies the same Jaccard-0.90 cross-split dedup against train to avoid
     near-duplicate contamination.
  4. Appends up to (MIN_N - current_count) new pairs per type.
  5. Rewrites data/splits_v2/test.jsonl and manifest.json with updated SHA-256
     and per_type stats.

Usage:
    python scripts/augment_test_v2.py [--min-n 20] [--seed 99]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import sys
from collections import Counter
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports (avoid loading all generators unless needed)
# ---------------------------------------------------------------------------

def _get_generators(seed: int) -> dict[str, object]:
    """Return a dict mapping error_type -> single-type generator instance."""
    from src.errors.tense_agreement import TenseAgreementErrorGenerator
    from src.errors.syntax_roles import CaseRoleErrorGenerator
    from src.errors.word_order import WordOrderErrorGenerator
    from src.errors.demonstrative_contraction import DemonstrativeContractionErrorGenerator
    from src.errors.morpheme_order import MorphemeOrderErrorGenerator
    from src.errors.quantifier_agreement import QuantifierAgreementErrorGenerator
    from src.errors.dialectal import DialectalParticipleErrorGenerator
    from src.errors.participle_swap import ParticipleSwapErrorGenerator
    from src.errors.punctuation_error import PunctuationErrorGenerator
    from src.errors.cross_clause_agreement import CrossClauseAgreementErrorGenerator
    from src.morphology.analyzer import MorphologicalAnalyzer
    analyzer = MorphologicalAnalyzer()
    return {
        "tense_agreement":         TenseAgreementErrorGenerator(error_rate=1.0, seed=seed, analyzer=analyzer),
        "case_role_preposition":   CaseRoleErrorGenerator(error_rate=1.0, seed=seed),
        "word_order":              WordOrderErrorGenerator(error_rate=1.0, seed=seed),
        "demonstrative_contraction": DemonstrativeContractionErrorGenerator(error_rate=1.0, seed=seed),
        "morpheme_order":          MorphemeOrderErrorGenerator(error_rate=1.0, seed=seed),
        "quantifier_agreement":    QuantifierAgreementErrorGenerator(error_rate=1.0, seed=seed),
        "dialectal_participle":    DialectalParticipleErrorGenerator(error_rate=1.0, seed=seed),
        "participle_voice_swap":   ParticipleSwapErrorGenerator(error_rate=1.0, seed=seed),
        "punctuation":             PunctuationErrorGenerator(error_rate=1.0, seed=seed),
        "cross_clause_agreement":  CrossClauseAgreementErrorGenerator(error_rate=1.0, seed=seed),
    }


# ---------------------------------------------------------------------------
# Jaccard trigram helpers (same as create_splits_v2.py)
# ---------------------------------------------------------------------------

def char_trigrams(text: str) -> frozenset[str]:
    t = text.strip()
    if len(t) < 3:
        return frozenset({t})
    return frozenset(t[i : i + 3] for i in range(len(t) - 2))


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def per_type_counts(records: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in records:
        for err in r.get("errors", []):
            et = err.get("type", err.get("error_type", "unknown"))
            counts[et] = counts.get(et, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Augment splits_v2 test for R10")
    ap.add_argument("--splits-dir",  default="data/splits",    help="Original splits directory")
    ap.add_argument("--v2-dir",      default="data/splits_v2", help="Splits_v2 directory to augment")
    ap.add_argument("--min-n",       type=int, default=20,     help="Minimum examples per type")
    ap.add_argument("--seed",        type=int, default=99,     help="RNG seed")
    args = ap.parse_args()

    splits_dir = Path(args.splits_dir)
    v2_dir     = Path(args.v2_dir)
    rng        = random.Random(args.seed)

    # ------------------------------------------------------------------
    # 1. Load current splits_v2
    # ------------------------------------------------------------------
    logger.info("Loading splits_v2 …")
    test_records  = _load_jsonl(v2_dir / "test.jsonl")
    train_records = _load_jsonl(v2_dir / "train.jsonl")

    # Build trigram index of train sources for Jaccard dedup
    logger.info("Building train trigram index (%d records) …", len(train_records))
    train_trigrams = [char_trigrams(r["source"]) for r in train_records]

    def is_near_duplicate_of_train(source: str, threshold: float = 0.90) -> bool:
        tg = char_trigrams(source)
        return any(jaccard(tg, t) >= threshold for t in train_trigrams)

    # ------------------------------------------------------------------
    # 2. Identify types that need augmentation
    # ------------------------------------------------------------------
    current_counts = per_type_counts(test_records)
    under_n = {
        t: args.min_n - count
        for t, count in current_counts.items()
        if count < args.min_n
    }
    # Also catch types with zero representation that generators could produce
    # (they won't be in current_counts at all)
    ZERO_TYPES = {"cross_clause_agreement"}
    for t in ZERO_TYPES:
        if t not in under_n:
            under_n[t] = args.min_n

    if not under_n:
        logger.info("All types already have >= %d examples. Nothing to do.", args.min_n)
        return

    logger.info(
        "Types needing augmentation: %s",
        {t: f"+{n}" for t, n in sorted(under_n.items(), key=lambda x: -x[1])}
    )

    # ------------------------------------------------------------------
    # 3. Source pool: original test sentences (split-boundary safe)
    # ------------------------------------------------------------------
    logger.info("Loading source pool from original test split …")
    orig_test = _load_jsonl(splits_dir / "test.jsonl")
    pool_sentences = list({r["original"] for r in orig_test})
    rng.shuffle(pool_sentences)
    logger.info("Source pool: %d unique clean sentences", len(pool_sentences))

    # ------------------------------------------------------------------
    # 4. Load generators for under-represented types only
    # ------------------------------------------------------------------
    logger.info("Initializing targeted generators …")
    all_generators = _get_generators(args.seed)
    generators = {t: g for t, g in all_generators.items() if t in under_n}

    # ------------------------------------------------------------------
    # 5. Generate targeted pairs
    # ------------------------------------------------------------------
    new_pairs: list[dict] = []
    counts_added: Counter = Counter()

    for error_type, generator in generators.items():
        needed = under_n[error_type]
        logger.info("  %s: need %d more pairs …", error_type, needed)
        added = 0
        for sentence in pool_sentences:
            if added >= needed:
                break
            try:
                result = generator.inject_errors(sentence)
            except Exception as exc:
                logger.debug("Generator %s failed on sentence: %s", error_type, exc)
                continue

            if not result.has_errors:
                continue
            errors = result.errors
            # Single-edit only
            if len(errors) != 1:
                continue
            # Verify the single error matches the target type
            actual_type = getattr(errors[0], "error_type", None) or getattr(errors[0], "type", error_type)
            if actual_type not in (error_type, ""):
                continue
            # Jaccard dedup vs train
            if is_near_duplicate_of_train(sentence):
                continue
            # Build JSONL record in the same schema as existing splits_v2
            err_obj = {
                "type":        error_type,
                "original":    getattr(errors[0], "original_span", ""),
                "error":       getattr(errors[0], "error_span", ""),
                "start":       getattr(errors[0], "start_pos", 0),
                "end":         getattr(errors[0], "end_pos", 0),
                "description": getattr(errors[0], "description", ""),
            }
            new_pairs.append({
                "original":  sentence,
                "corrupted": result.corrupted,
                "source":    result.corrupted,
                "target":    sentence,
                "errors":    [err_obj],
                "source_id": f"augment_r10_{error_type}_{added}",
                "category":  "augmented",
            })
            added += 1

        counts_added[error_type] = added
        logger.info("    -> added %d / %d for %s", added, needed, error_type)

    if not new_pairs:
        logger.warning("No new pairs generated. Under-represented types may have no eligible triggers in the source pool.")
        logger.warning("Consider expanding the source pool or accepting low counts for rare types.")
        return

    # ------------------------------------------------------------------
    # 6. Append and rewrite
    # ------------------------------------------------------------------
    test_records.extend(new_pairs)
    _write_jsonl(v2_dir / "test.jsonl", test_records)

    # ------------------------------------------------------------------
    # 7. Update manifest
    # ------------------------------------------------------------------
    with open(v2_dir / "manifest.json", encoding="utf-8") as f:
        manifest = json.load(f)

    n_trivial_new = sum(1 for r in test_records if r["source"] == r["target"])
    n_edited_new  = sum(1 for r in test_records if r["source"] != r["target"])

    manifest["splits"]["test"] = {
        "n_total":  len(test_records),
        "n_trivial": n_trivial_new,
        "n_edited":  n_edited_new,
        "sha256":   _sha256_file(v2_dir / "test.jsonl"),
    }

    new_counts = per_type_counts(test_records)
    manifest["per_type_test"] = new_counts
    manifest["under_20_error_types"] = {
        t: c for t, c in new_counts.items() if c < args.min_n
    }
    manifest["augmentation_r10"] = {
        "added_pairs": sum(counts_added.values()),
        "per_type_added": dict(counts_added),
        "source_pool": "data/splits/test.jsonl (original field, split-boundary safe)",
        "seed": args.seed,
    }

    manifest_sha = hashlib.sha256(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()
    manifest["_manifest_sha256"] = manifest_sha

    with open(v2_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    logger.info(
        "Done. Test: %d -> %d records (+%d). Updated manifest.",
        len(test_records) - len(new_pairs), len(test_records), len(new_pairs),
    )
    logger.info("Types still under %d: %s", args.min_n, manifest["under_20_error_types"])


if __name__ == "__main__":
    main()
