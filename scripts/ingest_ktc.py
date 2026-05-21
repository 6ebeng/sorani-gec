"""Ingest KTC (Kurdish Textbooks Corpus) files into the training corpus.

Reads every .txt file from data/ktc/<subject>/ subdirectories, extracts
Sorani Kurdish lines, and appends `category\tline` records to the
pre-resegment backup so resegment_clean_corpus.py can clean them.

Usage (from sorani-gec root):
    python scripts/ingest_ktc.py [--dry-run]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KTC_DIR = ROOT / "data" / "ktc"
CORPUS = ROOT / "data" / "clean" / "clean_corpus.txt"
BACKUP = CORPUS.with_suffix(CORPUS.suffix + ".pre_resegment.bak")

# Map KTC subdirectory name → canonical category label
_KTC_CAT: dict[str, str] = {
    "economy":      "economics",
    "genocide":     "history",
    "geography":    "geography",
    "history":      "history",
    "human-rights": "law",
    "kurdish":      "linguistics",
    "kurdology":    "linguistics",
    "philosophy":   "philosophy",
    "physics":      "sciences",
    "social-study": "social_sciences",
    "sociology":    "social_sciences",
    "theology":     "islamic_studies",
}

_MIN_LEN = 10   # characters; shorter lines are headers/footers


def _keep_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if len(s) < _MIN_LEN:
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not KTC_DIR.exists():
        print(f"ERROR: KTC directory not found: {KTC_DIR}", file=sys.stderr)
        sys.exit(1)

    if BACKUP.exists():
        existing_lines = BACKUP.read_text(encoding="utf-8").splitlines()
        existing_sentences: set[str] = {
            ln.split("\t", 1)[1].strip() if "\t" in ln else ln.strip()
            for ln in existing_lines if ln.strip()
        }
        print(f"Existing backup: {len(existing_lines):,} lines  "
              f"({len(existing_sentences):,} unique bodies)")
    else:
        existing_lines = CORPUS.read_text(encoding="utf-8").splitlines()
        existing_sentences = set()
        print("WARNING: no backup found — starting from clean_corpus.txt")

    new_records: list[str] = []
    skipped_dup = 0
    cat_counts: dict[str, int] = {}
    files_done = 0

    for subdir in sorted(KTC_DIR.iterdir()):
        if not subdir.is_dir() or subdir.name.startswith("."):
            continue
        cat = _KTC_CAT.get(subdir.name, subdir.name)
        for fpath in sorted(subdir.glob("*.txt")):
            try:
                text = fpath.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                print(f"  SKIP {fpath.name}: {exc}", file=sys.stderr)
                continue
            file_new = 0
            for line in text.splitlines():
                if not _keep_line(line):
                    continue
                body = line.strip()
                if body in existing_sentences:
                    skipped_dup += 1
                    continue
                existing_sentences.add(body)
                new_records.append(f"{cat}\t{body}")
                file_new += 1
            cat_counts[cat] = cat_counts.get(cat, 0) + file_new
            files_done += 1

    print(f"\nKTC files processed : {files_done}")
    print(f"New lines added     : {len(new_records):,}")
    print(f"Duplicates skipped  : {skipped_dup:,}")
    print("\nNew lines by category:")
    for cat, n in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {n:,}")

    if args.dry_run:
        print("\n[dry-run] No files written.")
        return

    combined = existing_lines + new_records
    combined_text = "\n".join(combined) + "\n"
    BACKUP.write_text(combined_text, encoding="utf-8")
    print(f"\nWrote {len(combined):,} total lines to {BACKUP}")
    CORPUS.write_text(combined_text, encoding="utf-8")
    print(f"Wrote {len(combined):,} total lines to {CORPUS}")
    print("\nNext step: python scripts/resegment_clean_corpus.py")


if __name__ == "__main__":
    main()
