"""Ingest OCR'd dissertation/book files into the training corpus.

Reads every .txt file from DISS_DIR, extracts Sorani Kurdish lines,
maps the filename-encoded category to a canonical label, and appends
`category\tline` records to the pre-resegment backup so the existing
resegment_clean_corpus.py pipeline can clean them identically to the
KTC source material.

Usage (from sorani-gec root):
    python scripts/ingest_dissertations.py [--dry-run]
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DISS_DIR = Path(r"c:\Users\Tishko\Desktop\Thesis\kurdish_books_and_dissertations\ocr_output_gemini")
CORPUS = ROOT / "data" / "clean" / "clean_corpus.txt"
BACKUP = CORPUS.with_suffix(CORPUS.suffix + ".pre_resegment.bak")

# ---------------------------------------------------------------------------
# Category normalisation
# ---------------------------------------------------------------------------
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
    # Sulaymaniyah thesis portal filenames embed a URL segment as the
    # second __ field; treat those as linguistics since almost all are
    # language/literature theses.
    "thesis.univsul.edu.iq": "linguistics",
    "general": "linguistics",   # unlabelled linguistics/grammar books
}

_URL_RX = re.compile(r"https?://|www\.|\.edu\.|\.ac\.")


def _extract_category(fname: str) -> str:
    """Return a canonical category from a dissertation filename."""
    stem = fname[:-4] if fname.endswith(".txt") else fname
    parts = stem.split("__")
    if len(parts) >= 3:
        raw = parts[1].strip().lower()
    elif len(parts) == 2:
        # institution__<Kurdish title>  — sulaymaniyah portal style
        # parts[1] is likely a URL fragment; classify as linguistics
        raw = parts[1].strip().lower()
    else:
        # No __ separator — standalone linguistics book
        raw = "general"
    return _CAT_MAP.get(raw, raw)


# ---------------------------------------------------------------------------
# Line-level filter (lightweight; full cleaning is done by resegmenter)
# ---------------------------------------------------------------------------
# Skip OCR file header comments
_HEADER_RX = re.compile(r"^\s*#")
# Skip Markdown-style heading lines from OCR output
_HEADING_RX = re.compile(r"^#+\s")
# Skip lines that are entirely ASCII / Latin (English abstracts, file paths)
_ALL_LATIN_RX = re.compile(r"^[A-Za-z0-9\s\-_./\\:,;\"\'()[\]{}@!?%&*+=<>~`|^#\n]*$")
# Skip lines shorter than 10 characters after stripping (titles, footers)
_MIN_LEN = 10


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
    if len(s) < _MIN_LEN:
        return False
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Print stats without writing anything")
    args = parser.parse_args()

    if not DISS_DIR.exists():
        print(f"ERROR: dissertation directory not found: {DISS_DIR}", file=sys.stderr)
        sys.exit(1)

    # Load existing lines from the backup (what we restore from)
    if BACKUP.exists():
        existing_lines = BACKUP.read_text(encoding="utf-8").splitlines()
        existing_sentences: set[str] = {
            ln.split("\t", 1)[1].strip() if "\t" in ln else ln.strip()
            for ln in existing_lines
            if ln.strip()
        }
        print(f"Existing backup: {len(existing_lines):,} lines  "
              f"({len(existing_sentences):,} unique sentence bodies)")
    else:
        print("WARNING: no .pre_resegment.bak found — will create new one from clean_corpus.txt")
        existing_lines = CORPUS.read_text(encoding="utf-8").splitlines()
        existing_sentences = set()

    new_records: list[str] = []
    skipped_duplicate = 0
    files_processed = 0
    cat_counts: dict[str, int] = {}

    for fname in sorted(os.listdir(DISS_DIR)):
        if not fname.endswith(".txt"):
            continue
        cat = _extract_category(fname)
        fpath = DISS_DIR / fname
        try:
            text = fpath.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            print(f"  SKIP {fname}: {exc}", file=sys.stderr)
            continue

        file_new = 0
        for line in text.splitlines():
            if not _keep_line(line):
                continue
            body = line.strip()
            if body in existing_sentences:
                skipped_duplicate += 1
                continue
            existing_sentences.add(body)   # prevent intra-run duplication
            new_records.append(f"{cat}\t{body}")
            file_new += 1

        cat_counts[cat] = cat_counts.get(cat, 0) + file_new
        files_processed += 1

    print(f"\nFiles processed : {files_processed}")
    print(f"New lines added : {len(new_records):,}")
    print(f"Duplicates skip : {skipped_duplicate:,}")
    print("\nNew lines by category:")
    for cat, n in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {n:,}")

    if args.dry_run:
        print("\n[dry-run] No files written.")
        return

    # Append to backup (which is what we restore from before each resegment run)
    combined = existing_lines + new_records
    combined_text = "\n".join(combined) + "\n"
    BACKUP.write_text(combined_text, encoding="utf-8")
    print(f"\nWrote {len(combined):,} total lines to {BACKUP}")

    # Also update clean_corpus.txt so the resegmenter sees the new content
    CORPUS.write_text(combined_text, encoding="utf-8")
    print(f"Wrote {len(combined):,} total lines to {CORPUS}")
    print("\nNext step: python scripts/resegment_clean_corpus.py")


if __name__ == "__main__":
    main()
