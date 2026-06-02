"""
Phase C: Create splits_v2 — single-edit pairs, cross-split Jaccard dedup, SHA-256 manifest.

Implements R3 (retire 1,003-pair test as headline; regenerate clean splits),
R29 (enforce single-edit per pair), and R34 (cross-split Jaccard-0.90 dedup).

Usage:
    cd Implementation/sorani-gec
    python scripts/create_splits_v2.py [--input-dir data/splits] [--output-dir data/splits_v2]
"""

import argparse
import hashlib
import json
import logging
import re
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Jaccard similarity (character trigrams)
# ---------------------------------------------------------------------------

def char_trigrams(text: str) -> frozenset[str]:
    """Return the set of character trigrams for *text*."""
    t = text.strip()
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


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_split(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_split(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def filter_single_edit(records: list[dict]) -> tuple[list[dict], dict]:
    """Keep only trivial (0 errors) and single-edit (1 error) pairs.

    Returns (kept, stats).
    """
    kept = []
    stats = {"kept_trivial": 0, "kept_single": 0, "dropped_multi": 0}
    for r in records:
        n_errors = len(r.get("errors", []))
        if n_errors == 0:
            stats["kept_trivial"] += 1
            kept.append(r)
        elif n_errors == 1:
            stats["kept_single"] += 1
            kept.append(r)
        else:
            stats["dropped_multi"] += 1
    return kept, stats


def cross_split_dedup(
    anchor: list[dict],
    candidates: list[dict],
    threshold: float = 0.90,
    key: str = "original",
) -> tuple[list[dict], int]:
    """Remove from *candidates* any record whose *key* field has Jaccard similarity
    ≥ *threshold* with any record in *anchor*.

    Returns (filtered_candidates, n_removed).
    """
    logger.info(
        "Cross-split dedup: building trigram index for %d anchor records …", len(anchor)
    )
    anchor_tgrams = [char_trigrams(r.get(key, "")) for r in anchor]

    kept = []
    removed = 0
    for r in candidates:
        c_tg = char_trigrams(r.get(key, ""))
        is_near_dup = any(jaccard(c_tg, a_tg) >= threshold for a_tg in anchor_tgrams)
        if is_near_dup:
            removed += 1
        else:
            kept.append(r)

    return kept, removed


def per_type_counts(records: list[dict]) -> dict[str, int]:
    # Error objects use key "type" (not "error_type")
    counts: dict[str, int] = {}
    for r in records:
        for err in r.get("errors", []):
            et = err.get("type", err.get("error_type", "unknown"))
            counts[et] = counts.get(et, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


# ---------------------------------------------------------------------------
# Provenance + residual-overlap helpers (audit rows 2.3/2.4)
# ---------------------------------------------------------------------------

def load_provenance(path: Path) -> dict[str, tuple[str, str]]:
    """Map clean sentence -> (source_id, category) from synthetic annotations."""
    prov: dict[str, tuple[str, str]] = {}
    if not path.exists():
        logger.warning("Provenance file %s not found; records left un-enriched.", path)
        return prov
    for r in load_split(path):
        prov[r.get("original", "").strip()] = (
            str(r.get("source_id", "")), r.get("category", "")
        )
    return prov


def enrich_provenance(records: list[dict], prov: dict[str, tuple[str, str]]) -> int:
    """Attach source_id/category to each record in place. Returns match count."""
    matched = 0
    for r in records:
        key = r.get("original", "").strip()
        if key in prov:
            sid, cat = prov[key]
            r["source_id"] = sid
            r["category"] = cat
            matched += 1
    return matched


def source_id_overlap(anchor: list[dict], candidates: list[dict]) -> int:
    """Count candidate records whose source_id also appears in anchor."""
    anchor_ids = {r.get("source_id") for r in anchor if r.get("source_id")}
    return sum(1 for r in candidates if r.get("source_id") in anchor_ids)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Create splits_v2 from splits")
    parser.add_argument("--input-dir", default="data/splits")
    parser.add_argument("--output-dir", default="data/splits_v2")
    parser.add_argument(
        "--jaccard-threshold", type=float, default=0.90,
        help="Jaccard similarity threshold for cross-split near-dup removal (default 0.90)"
    )
    parser.add_argument(
        "--keep-trivial", action="store_true", default=True,
        help="Retain trivial (source==target) pairs as a copy-through signal"
    )
    parser.add_argument(
        "--drop-trivial", dest="keep_trivial", action="store_false",
        help="Drop trivial (source==target) pairs — use for edited-subset-only splits"
    )
    parser.add_argument(
        "--dedup-dev-test", action="store_true", default=False,
        help="Also remove test pairs that are near-duplicates of dev (L6-02); "
             "off by default so splits_v2 reproduces bit-for-bit."
    )
    parser.add_argument(
        "--provenance", default=None,
        help="Path to synthetic annotations jsonl; when given, joins source_id/category "
             "onto each record and reports residual cross-split source overlap."
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    threshold = args.jaccard_threshold

    # ------------------------------------------------------------------
    # 1. Load original splits
    # ------------------------------------------------------------------
    logger.info("Loading original splits from %s …", input_dir)
    orig_train = load_split(input_dir / "train.jsonl")
    orig_dev   = load_split(input_dir / "dev.jsonl")
    orig_test  = load_split(input_dir / "test.jsonl")
    logger.info(
        "Loaded: train=%d  dev=%d  test=%d",
        len(orig_train), len(orig_dev), len(orig_test),
    )

    # ------------------------------------------------------------------
    # 2. R29: Single-edit filter
    # ------------------------------------------------------------------
    logger.info("R29: Filtering to single-edit pairs …")
    train_se, s_train = filter_single_edit(orig_train)
    dev_se,   s_dev   = filter_single_edit(orig_dev)
    test_se,  s_test  = filter_single_edit(orig_test)

    for name, orig, filt, stats in [
        ("train", orig_train, train_se, s_train),
        ("dev",   orig_dev,   dev_se,   s_dev),
        ("test",  orig_test,  test_se,  s_test),
    ]:
        logger.info(
            "%s: %d → %d  (dropped %d multi-edit; trivial=%d, single=%d)",
            name, len(orig), len(filt),
            stats["dropped_multi"], stats["kept_trivial"], stats["kept_single"],
        )

    # Optionally drop trivial pairs
    if not args.keep_trivial:
        logger.info("--drop-trivial: removing trivial (source==target) pairs")
        train_se = [r for r in train_se if r.get("errors", [])]
        dev_se   = [r for r in dev_se   if r.get("errors", [])]
        test_se  = [r for r in test_se  if r.get("errors", [])]
        logger.info(
            "After trivial drop: train=%d  dev=%d  test=%d",
            len(train_se), len(dev_se), len(test_se),
        )

    # ------------------------------------------------------------------
    # 3. R34: Cross-split Jaccard dedup (dev, test vs train)
    # ------------------------------------------------------------------
    logger.info("R34: Cross-split Jaccard-%.2f dedup …", threshold)

    dev_dedup,  dev_removed  = cross_split_dedup(train_se, dev_se,  threshold)
    test_dedup, test_removed = cross_split_dedup(train_se, test_se, threshold)

    logger.info("Dev:  %d → %d  (%d removed as near-dup of train)", len(dev_se),  len(dev_dedup),  dev_removed)
    logger.info("Test: %d → %d  (%d removed as near-dup of train)", len(test_se), len(test_dedup), test_removed)

    # ------------------------------------------------------------------
    # 3b. L6-02: optional dev↔test near-dup dedup (off by default)
    # ------------------------------------------------------------------
    dev_test_removed = 0
    if args.dedup_dev_test:
        logger.info("L6-02: dev↔test Jaccard-%.2f dedup (removing test near-dups of dev) …", threshold)
        test_dedup, dev_test_removed = cross_split_dedup(dev_dedup, test_dedup, threshold)
        logger.info("Test: → %d  (%d removed as near-dup of dev)", len(test_dedup), dev_test_removed)

    # ------------------------------------------------------------------
    # 3c. Optional provenance join + residual cross-split overlap (rows 2.3/2.4)
    # ------------------------------------------------------------------
    residual = {}
    if args.provenance:
        prov = load_provenance(Path(args.provenance))
        m_tr = enrich_provenance(train_se,   prov)
        m_dv = enrich_provenance(dev_dedup,  prov)
        m_te = enrich_provenance(test_dedup, prov)
        residual = {
            "provenance_matched": {"train": m_tr, "dev": m_dv, "test": m_te},
            "test_source_in_train": source_id_overlap(train_se, test_dedup),
            "dev_source_in_train":  source_id_overlap(train_se, dev_dedup),
            "dev_source_in_test":   source_id_overlap(test_dedup, dev_dedup),
        }
        logger.info(
            "Provenance matched train=%d dev=%d test=%d; residual source overlap "
            "test∈train=%d dev∈train=%d dev∈test=%d",
            m_tr, m_dv, m_te,
            residual["test_source_in_train"],
            residual["dev_source_in_train"],
            residual["dev_source_in_test"],
        )

    # ------------------------------------------------------------------
    # 4. Per-type statistics (for R10 gap analysis)
    # ------------------------------------------------------------------
    test_type_counts = per_type_counts(test_dedup)
    logger.info("=== Per-type counts in test_v2 (edited pairs only) ===")
    for et, cnt in test_type_counts.items():
        flag = "  *** n<20" if cnt < 20 else ""
        logger.info("  %-40s %4d%s", et, cnt, flag)

    under_20 = {et: cnt for et, cnt in test_type_counts.items() if cnt < 20}
    if under_20:
        logger.warning(
            "R10: %d error type(s) have n < 20 in test_v2; augmentation needed: %s",
            len(under_20),
            list(under_20.keys()),
        )

    # ------------------------------------------------------------------
    # 5. Save splits_v2
    # ------------------------------------------------------------------
    logger.info("Saving splits_v2 to %s …", output_dir)
    save_split(train_se,   output_dir / "train.jsonl")
    save_split(dev_dedup,  output_dir / "dev.jsonl")
    save_split(test_dedup, output_dir / "test.jsonl")

    # ------------------------------------------------------------------
    # 6. SHA-256 manifest (exit gate for Phase C)
    # ------------------------------------------------------------------
    manifest = {
        "phase": "C",
        "description": (
            "splits_v2: single-edit-only filter (R29) applied; "
            f"cross-split Jaccard-{threshold:.2f} dedup applied (R34); "
            "trivial (source==target) pairs retained as copy-through signal."
        ),
        "splits": {},
        "per_type_test": test_type_counts,
        "under_20_error_types": under_20,
        "dev_test_dedup_applied": bool(args.dedup_dev_test),
        "dev_test_near_dups_removed": dev_test_removed,
        "residual_overlap": residual,
    }

    for name in ("train", "dev", "test"):
        p = output_dir / f"{name}.jsonl"
        records = load_split(p)
        n_trivial = sum(1 for r in records if not r.get("errors", []))
        n_edited  = len(records) - n_trivial
        manifest["splits"][name] = {
            "n_total":   len(records),
            "n_trivial": n_trivial,
            "n_edited":  n_edited,
            "sha256":    sha256_file(p),
        }
        logger.info(
            "%s  total=%d  trivial=%d (%.1f%%)  edited=%d  sha256=%s…",
            name, len(records), n_trivial,
            100 * n_trivial / max(len(records), 1),
            n_edited,
            sha256_file(p)[:16],
        )

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    manifest["_manifest_sha256"] = sha256_file(manifest_path)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    logger.info("Manifest written to %s", manifest_path)
    logger.info("Phase C exit gate: splits_v2/ created with SHA-256 manifest.")


if __name__ == "__main__":
    main()
