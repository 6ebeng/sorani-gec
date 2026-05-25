"""Collect natural Sorani Kurdish sentences for the R31 natural test set.

Draws from two locally-available authoritative corpora:
  1. OCR'd university dissertations — Koya, Sulaymaniyah, Salahaddin
     universities (C:\\...\\kurdish_books_and_dissertations\\ocr_output_gemini\\)
  2. KTC (Kurdish Textbooks Corpus) subject files (data/ktc/)

Sentences extracted from these human-authored academic texts may contain
naturally-occurring grammatical errors (the kind a real GEC system must fix),
as opposed to the synthetically-injected errors in the training pipeline.

Selection heuristic:
  - Sentences where the normaliser modifies the text are prioritised, since
    a divergence between raw and normalised form suggests a potential surface
    error (orthographic confusion, wrong character variant, etc.).
  - Sentences that pass unchanged through normalisation are also included to
    give annotators a mix of erroneous and clean examples.
  - Results are stratified across source institutions and KTC subject domains.

Output is appended to data/natural_test/sentences.jsonl with target_text=""
so a human annotator can fill in corrections and error_types.

Usage:
    # Default: 200 new sentences from dissertations + KTC combined
    python scripts/13_collect_natural_test_sentences.py

    # Dissertations only, 100 sentences
    python scripts/13_collect_natural_test_sentences.py --source dissertations --count 100

    # KTC only, 150 sentences
    python scripts/13_collect_natural_test_sentences.py --source ktc --count 150

    # Dry-run: show what would be collected without writing
    python scripts/13_collect_natural_test_sentences.py --dry-run --count 30
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from src.data.normalizer import normalize_sorani
except ImportError:
    def normalize_sorani(text: str) -> str:  # type: ignore[misc]
        return text

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "natural_test"
OUT_FILE = OUT_DIR / "sentences.jsonl"

# Dissertation OCR output directory (Koya, Sulaymaniyah, Salahaddin universities)
DISS_DIR = Path(r"C:\Users\Tishko\Desktop\Thesis\kurdish_books_and_dissertations\ocr_output_gemini")

# KTC corpus directory (subject subdirectories)
KTC_DIR = ROOT / "data" / "ktc"

# KTC subdirectory → canonical category
_KTC_CAT: dict[str, str] = {
    "economy": "economics",
    "genocide": "history",
    "geography": "geography",
    "history": "history",
    "human-rights": "law",
    "kurdish": "linguistics",
    "kurdology": "linguistics",
    "philosophy": "philosophy",
    "physics": "sciences",
    "social-study": "social_sciences",
    "sociology": "social_sciences",
    "theology": "islamic_studies",
}

# University tag extracted from dissertation filenames  (e.g. "koya_university")
_UNIV_RX = re.compile(r"^(koya_university|sulaymaniyah|salahaddin|sulaymanyah|sulaymani)", re.I)

_CAT_MAP: dict[str, str] = {
    "lingustics": "linguistics",
    "linguistics": "linguistics",
    "litreture": "literature",
    "literature": "literature",
    "psycology": "psychology",
    "psychology": "psychology",
    "geography": "geography",
    "history": "history",
    "archaeology": "archaeology",
    "media": "media",
    "politics": "politics",
    "economics": "economics",
    "law": "law",
    "sciences": "sciences",
    "social_sciences": "social_sciences",
    "islamic_studies": "islamic_studies",
    "general": "linguistics",
}

# Line-level filters (reuse same rules as ingest_dissertations.py)
_HEADER_RX = re.compile(r"^\s*#")
_HEADING_RX = re.compile(r"^#+\s")
_ALL_LATIN_RX = re.compile(r"^[A-Za-z0-9\s\-_./\\:,;\"\'()[\]{}@!?%&*+=<>~`|^#\n]*$")
_KU_CHARS = re.compile(r"[\u0600-\u06FF\u0750-\u077F]")
_MIN_LEN = 40    # longer minimum for test set (want full sentences, not fragments)
_MAX_LEN = 500


def _keep_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if _HEADER_RX.match(s):
        return False
    if _HEADING_RX.match(s):
        return False
    if _ALL_LATIN_RX.match(s):
        return False
    if not _KU_CHARS.search(s):
        return False
    if len(s) < _MIN_LEN or len(s) > _MAX_LEN:
        return False
    return True


def _diss_category(stem: str) -> str:
    parts = stem.split("__")
    raw = parts[1].strip().lower() if len(parts) >= 3 else "general"
    return _CAT_MAP.get(raw, "general")


def _diss_source_tag(stem: str) -> str:
    parts = stem.split("__")
    institution = parts[0].strip() if parts else "unknown"
    return f"dissertation:{institution}"


def _load_existing_texts(path: Path) -> set[str]:
    seen: set[str] = set()
    if not path.exists():
        return seen
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    seen.add(json.loads(line).get("source_text", "").strip())
                except json.JSONDecodeError:
                    pass
    return seen


def _collect_from_dissertations(
    existing: set[str],
    rng: random.Random,
) -> list[dict]:
    """Extract sentences from all OCR'd dissertation .txt files."""
    if not DISS_DIR.exists():
        logger.error("Dissertation directory not found: %s", DISS_DIR)
        return []

    candidates: list[dict] = []
    txt_files = sorted(DISS_DIR.glob("*.txt"))
    logger.info("Dissertation .txt files found: %d", len(txt_files))

    for txt_path in txt_files:
        stem = txt_path.stem
        category = _diss_category(stem)
        source_tag = _diss_source_tag(stem)
        try:
            raw_text = txt_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            logger.warning("Cannot read %s: %s", txt_path.name, exc)
            continue

        for raw_line in raw_text.splitlines():
            if not _keep_line(raw_line):
                continue
            norm = normalize_sorani(raw_line.strip())
            if norm in existing:
                continue
            # Flag whether normaliser changed anything (potential error signal)
            modified_by_norm = norm != raw_line.strip()
            candidates.append(
                {
                    "source_text": norm,
                    "source_url": source_tag,
                    "register": "learner",
                    "dialect": "central",
                    "category": category,
                    "_modified": modified_by_norm,
                    "_stem": stem,
                }
            )

    logger.info("Dissertation candidates (after dedup): %d", len(candidates))
    return candidates


def _collect_from_ktc(
    existing: set[str],
    rng: random.Random,
) -> list[dict]:
    """Extract sentences from KTC subject .txt files."""
    if not KTC_DIR.exists():
        logger.error("KTC directory not found: %s", KTC_DIR)
        return []

    candidates: list[dict] = []
    for subdir in sorted(KTC_DIR.iterdir()):
        if not subdir.is_dir() or subdir.name.startswith("."):
            continue
        cat = _KTC_CAT.get(subdir.name, subdir.name)
        for txt_path in sorted(subdir.glob("*.txt")):
            try:
                raw_text = txt_path.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                logger.warning("Cannot read %s: %s", txt_path.name, exc)
                continue
            for raw_line in raw_text.splitlines():
                if not _keep_line(raw_line):
                    continue
                norm = normalize_sorani(raw_line.strip())
                if norm in existing:
                    continue
                modified_by_norm = norm != raw_line.strip()
                candidates.append(
                    {
                        "source_text": norm,
                        "source_url": f"ktc:{subdir.name}/{txt_path.stem}",
                        "register": "blog",
                        "dialect": "central",
                        "category": cat,
                        "_modified": modified_by_norm,
                        "_stem": txt_path.stem,
                    }
                )

    logger.info("KTC candidates (after dedup): %d", len(candidates))
    return candidates


def _stratified_sample(
    candidates: list[dict],
    n: int,
    rng: random.Random,
    prefer_modified: bool = True,
) -> list[dict]:
    """Sample n records, stratifying by category; slightly prefer norm-modified ones."""
    # Split into modified (potential errors) and unmodified
    modified = [c for c in candidates if c["_modified"]]
    unmodified = [c for c in candidates if not c["_modified"]]

    # Aim for ~40% modified (natural error candidates), ~60% clean
    n_modified = min(len(modified), round(n * 0.40)) if prefer_modified else 0
    n_unmodified = min(len(unmodified), n - n_modified)
    actual_n = n_modified + n_unmodified

    if actual_n < n:
        logger.warning("Only %d candidates available (requested %d)", actual_n, n)

    # Stratify modified by category
    def _strat(pool: list[dict], quota: int) -> list[dict]:
        buckets: dict[str, list[dict]] = defaultdict(list)
        for item in pool:
            buckets[item["category"]].append(item)
        out: list[dict] = []
        remainder: list[dict] = []
        for cat, items in sorted(buckets.items()):
            q = max(1, round(len(items) / len(pool) * quota)) if pool else 0
            rng.shuffle(items)
            out.extend(items[:q])
            remainder.extend(items[q:])
        rng.shuffle(remainder)
        if len(out) < quota:
            out.extend(remainder[: quota - len(out)])
        return out[:quota]

    rng.shuffle(modified)
    rng.shuffle(unmodified)
    selected = _strat(modified, n_modified) + _strat(unmodified, n_unmodified)
    rng.shuffle(selected)
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect natural Sorani sentences from dissertations and KTC for R31 test set.",
    )
    parser.add_argument(
        "--source",
        choices=["dissertations", "ktc", "both"],
        default="both",
        help="Which corpus to draw from (default: both).",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=200,
        help="Number of new sentences to collect (default: 200).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for stratified sampling.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print collected records without writing to disk.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUT_FILE,
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    existing = _load_existing_texts(args.output)
    logger.info("Existing natural-test entries: %d", len(existing))

    candidates: list[dict] = []
    if args.source in ("dissertations", "both"):
        candidates.extend(_collect_from_dissertations(existing, rng))
    if args.source in ("ktc", "both"):
        candidates.extend(_collect_from_ktc(existing, rng))

    if not candidates:
        logger.error("No candidates found. Check that source directories exist.")
        return 1

    selected = _stratified_sample(candidates, args.count, rng)

    out_handle = None if args.dry_run else args.output.open("a", encoding="utf-8")
    written = 0
    try:
        for i, item in enumerate(selected, 1):
            record = {
                "id": f"nat_{item['_stem'][:12]}_{i:04d}",
                "source_text": item["source_text"],
                "target_text": "",          # Fill by hand after review
                "source_url": item["source_url"],
                "register": item["register"],
                "dialect": item["dialect"],
                "error_types": [],           # Tag after annotation
                "annotator_ids": [],
                "notes": (
                    "norm_modified=true — normaliser changed this sentence; "
                    "check for orthographic or character-variant errors."
                    if item["_modified"] else
                    "norm_unchanged — sentence may still contain agreement/morphological errors; "
                    "review manually."
                ),
            }
            if args.dry_run:
                print(json.dumps(record, ensure_ascii=False))
            else:
                out_handle.write(json.dumps(record, ensure_ascii=False) + "\n")  # type: ignore[union-attr]
                written += 1
    finally:
        if out_handle:
            out_handle.close()

    logger.info(
        "%s %d records to %s",
        "Would write" if args.dry_run else "Wrote",
        len(selected) if args.dry_run else written,
        args.output if not args.dry_run else "(dry-run)",
    )
    logger.info(
        "Next: open %s, review each entry, fill target_text + error_types, "
        "then run: python scripts/build_m2_from_jsonl.py "
        "--input data/natural_test/sentences.jsonl "
        "--output data/natural_test/annotations.m2",
        args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
