"""
Step 14: Build a scaled single-edit training split for the positive-result retrain.

Why this exists
---------------
The released splits_v2 train set holds only 5,253 pairs because the R29
single-edit constraint plus cross-split dedup shrank an originally larger pool.
A 300M-parameter ByT5 starves on that. This script lifts the *train* set
roughly 5-7x while keeping every methodological guarantee that splits_v2 made:

  * single-edit discipline (R29): keep only 0-error (trivial) or 1-error pairs;
  * no train/eval leakage: drop any train pair whose clean side is a trigram
    near-duplicate (Jaccard >= threshold) of a dev OR test sentence;
  * the dev and test splits are copied byte-for-byte from splits_v2, so every
    published number and the agreement-density analysis stay comparable.

Only the training distribution changes. The held-out sets do not.

Optionally over-samples the core-agreement error types so the morphological
pathway has more agreement examples to learn from (a train-side choice, not
test contamination).

Usage
-----
    cd Implementation/sorani-gec
    python scripts/14_build_scaled_train.py \
        --pool-input data/balanced/balanced_corpus.txt \
        --target 50000 \
        --eval-dir data/splits_v2 \
        --output-dir data/splits_scaled \
        --agreement-oversample 1.5

Smoke (tiny, fast):
    python scripts/14_build_scaled_train.py --target 800 \
        --pool-out data/synthetic_smoke --output-dir data/splits_smoke
"""

import argparse
import hashlib
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Core-agreement generator types (mirror scripts/13_agreement_subset_rescore.py).
CORE_AGREEMENT_TYPES = {
    "subject_verb_number",
    "subject_verb",
    "clitic_form",
    "clitic",
    "possessive_clitic",
    "noun_adjective_agreement",
    "case_role_preposition",
    "tense_agreement",
    "quantifier_agreement",
    "cross_clause_agreement",
    "conditional_agreement",
    "negative_concord",
    "vocative_imperative",
    "ergative",
}


# --- helpers ---------------------------------------------------------------

def char_trigrams(text: str) -> frozenset:
    t = (text or "").strip()
    if len(t) < 3:
        return frozenset([t]) if t else frozenset()
    return frozenset(t[i : i + 3] for i in range(len(t) - 2))


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def load_jsonl(path: Path) -> list:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_jsonl(rows: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def primary_error_type(row: dict):
    errors = row.get("errors") or []
    if not errors:
        return None
    return errors[0].get("type")


def clean_key(row: dict) -> str:
    """The clean (target) side, used for dedup and exact-dup keys."""
    return (row.get("original") or row.get("target") or "").strip()


def sha256_rows(rows: list) -> str:
    h = hashlib.sha256()
    for r in sorted(clean_key(r) for r in rows):
        h.update(r.encode("utf-8"))
    return h.hexdigest()


# --- pool generation -------------------------------------------------------

def generate_pool(pool_input: Path, pool_out: Path, target: int,
                  corruption_ratio: float, seed: int) -> Path:
    """Run the error pipeline to produce a large annotations.jsonl pool."""
    from src.errors.pipeline import ErrorPipeline

    if not pool_input.exists():
        logger.error("Pool input corpus not found: %s", pool_input)
        raise SystemExit(1)

    pool_out.mkdir(parents=True, exist_ok=True)
    logger.info("Generating ~%d pairs from %s (corruption_ratio=%.2f) ...",
                target, pool_input, corruption_ratio)
    pipeline = ErrorPipeline(error_rate=0.15, seed=seed)
    stats = pipeline.process_corpus(
        input_file=str(pool_input),
        output_dir=str(pool_out),
        target_pairs=target,
        corruption_ratio=corruption_ratio,
        validate_errors=True,
    )
    logger.info("Pool generation stats: %s", stats)
    return pool_out / "annotations.jsonl"


# --- main ------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Build a scaled single-edit train split")
    p.add_argument("--pool-input", default="data/balanced/balanced_corpus.txt",
                   help="Clean-sentence corpus to corrupt (one sentence per line)")
    p.add_argument("--pool-out", default="data/synthetic_scaled",
                   help="Directory for the generated annotations pool")
    p.add_argument("--target", type=int, default=50000,
                   help="Target number of pairs to generate for the pool")
    p.add_argument("--corruption-ratio", type=float, default=0.7)
    p.add_argument("--eval-dir", default="data/splits_v2",
                   help="Source of the FIXED dev/test splits (copied verbatim)")
    p.add_argument("--output-dir", default="data/splits_scaled")
    p.add_argument("--jaccard-threshold", type=float, default=0.90,
                   help="Drop train pairs whose clean side is >= this similar to dev/test")
    p.add_argument("--agreement-oversample", type=float, default=1.0,
                   help="Replication factor for core-agreement single-edit train pairs "
                        "(1.0 = no oversampling; 2.0 = duplicate them once)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--regenerate", action="store_true", default=False,
                   help="Force regeneration of the pool even if annotations.jsonl exists")
    args = p.parse_args()

    eval_dir = Path(args.eval_dir)
    out_dir = Path(args.output_dir)
    pool_out = Path(args.pool_out)
    pool_annotations = pool_out / "annotations.jsonl"

    dev_path = eval_dir / "dev.jsonl"
    test_path = eval_dir / "test.jsonl"
    for path in (dev_path, test_path):
        if not path.exists():
            logger.error("Fixed eval split missing: %s", path)
            raise SystemExit(1)

    # 1) Generate (or reuse) the large pool.
    if args.regenerate or not pool_annotations.exists():
        pool_annotations = generate_pool(
            Path(args.pool_input), pool_out, args.target,
            args.corruption_ratio, args.seed,
        )
    else:
        logger.info("Reusing existing pool: %s (use --regenerate to rebuild)",
                    pool_annotations)

    pool = load_jsonl(pool_annotations)
    logger.info("Loaded pool: %d pairs", len(pool))

    # 2) Single-edit filter (R29): keep 0- or 1-error pairs only.
    single = [r for r in pool if len(r.get("errors", [])) <= 1]
    n_trivial = sum(1 for r in single if len(r.get("errors", [])) == 0)
    logger.info("Single-edit filter: kept %d (%d trivial, %d single)",
                len(single), n_trivial, len(single) - n_trivial)

    # 3) Build the dev+test trigram index and drop near-duplicate train pairs.
    dev = load_jsonl(dev_path)
    test = load_jsonl(test_path)
    eval_rows = dev + test
    eval_tgrams = [char_trigrams(clean_key(r)) for r in eval_rows]
    eval_exact = {clean_key(r) for r in eval_rows}
    logger.info("Built eval index: %d dev + %d test = %d sentences",
                len(dev), len(test), len(eval_rows))

    kept = []
    seen_clean = set()
    n_leak, n_dup = 0, 0
    for r in single:
        key = clean_key(r)
        if not key:
            continue
        if key in eval_exact:
            n_leak += 1
            continue
        if key in seen_clean:
            n_dup += 1
            continue
        c_tg = char_trigrams(key)
        if any(jaccard(c_tg, e_tg) >= args.jaccard_threshold for e_tg in eval_tgrams):
            n_leak += 1
            continue
        seen_clean.add(key)
        kept.append(r)
    logger.info("Dedup vs eval: dropped %d leak/near-dup, %d intra-train dup; kept %d",
                n_leak, n_dup, len(kept))

    # 4) Optional agreement oversampling (train-side only).
    train = list(kept)
    if args.agreement_oversample > 1.0:
        import random as _random
        rng = _random.Random(args.seed)
        agr = [r for r in kept if primary_error_type(r) in CORE_AGREEMENT_TYPES]
        factor = args.agreement_oversample
        whole = int(factor) - 1          # guaranteed extra full copies
        frac = factor - int(factor)      # fractional extra copy probability
        added = agr * whole
        if frac > 0.0:
            added += [r for r in agr if rng.random() < frac]
        train = kept + added
        logger.info("Agreement oversample x%.2f: +%d core-agreement copies (%d unique agr)",
                    factor, len(added), len(agr))

    # 5) Write train.jsonl; copy dev/test byte-for-byte from the eval dir.
    import shutil
    out_dir.mkdir(parents=True, exist_ok=True)
    save_jsonl(train, out_dir / "train.jsonl")
    shutil.copyfile(dev_path, out_dir / "dev.jsonl")
    shutil.copyfile(test_path, out_dir / "test.jsonl")

    def file_sha256(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    # 6) Manifest with provenance + a leakage guarantee record.
    train_types: dict = {}
    for r in train:
        et = primary_error_type(r) or "trivial"
        train_types[et] = train_types.get(et, 0) + 1
    manifest = {
        "built_from_pool": str(pool_annotations),
        "pool_size": len(pool),
        "single_edit_kept": len(single),
        "train_size": len(train),
        "train_unique": len(kept),
        "dev_size": len(dev),
        "test_size": len(test),
        "dropped_leak_near_dup": n_leak,
        "dropped_intra_train_dup": n_dup,
        "jaccard_threshold": args.jaccard_threshold,
        "agreement_oversample": args.agreement_oversample,
        "seed": args.seed,
        "train_sha256": sha256_rows(train),
        "dev_sha256": sha256_rows(dev),
        "test_sha256": sha256_rows(test),
        "test_matches_splits_v2": file_sha256(out_dir / "test.jsonl") == file_sha256(test_path),
        "dev_matches_splits_v2": file_sha256(out_dir / "dev.jsonl") == file_sha256(dev_path),
        "train_error_type_counts": dict(sorted(train_types.items(), key=lambda kv: -kv[1])),
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    logger.info("=" * 64)
    logger.info("Wrote scaled splits to %s", out_dir)
    logger.info("  train=%d  dev=%d  test=%d", len(train), len(dev), len(test))
    logger.info("  test byte-identical to splits_v2: %s", manifest["test_matches_splits_v2"])
    logger.info("  manifest: %s", out_dir / "manifest.json")
    logger.info("=" * 64)


if __name__ == "__main__":
    main()
