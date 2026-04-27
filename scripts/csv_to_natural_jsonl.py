"""Convert a reviewed natural-test CSV into the canonical JSONL.

Reads `data/natural_test/sentences_for_review.csv` (UTF-8 BOM, Excel-friendly),
keeps rows where `keep` is empty / `y` / `edit`, drops rows marked `n`,
and writes one JSON object per line to `data/natural_test/sentences.jsonl`.

Usage:
    python scripts/csv_to_natural_jsonl.py
    python scripts/csv_to_natural_jsonl.py --input my_reviewed.csv \
        --output data/natural_test/sentences.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN = ROOT / "data" / "natural_test" / "sentences_for_review.csv"
DEFAULT_OUT = ROOT / "data" / "natural_test" / "sentences.jsonl"

VALID_REGISTERS = {"news", "social", "blog", "learner", "chat", "forum", "other"}
VALID_DIALECTS = {"central", "northern", "southern", "mixed", "unknown"}


def _split(value: str) -> list[str]:
    if not value:
        return []
    return [v.strip() for v in value.split(";") if v.strip()]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=DEFAULT_IN)
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Not found: {args.input}")

    written = 0
    dropped = 0
    skipped = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.input.open(encoding="utf-8-sig", newline="") as fin, \
         args.output.open("w", encoding="utf-8") as fout:
        reader = csv.DictReader(fin)
        for row in reader:
            keep = (row.get("keep") or "").strip().lower()
            if keep == "n":
                dropped += 1
                continue
            src = (row.get("source_text") or "").strip()
            tgt = (row.get("target_text") or "").strip()
            if not src or not tgt:
                skipped += 1
                continue

            register = (row.get("register") or "other").strip().lower()
            if register not in VALID_REGISTERS:
                register = "other"
            dialect = (row.get("dialect") or "unknown").strip().lower()
            if dialect not in VALID_DIALECTS:
                dialect = "unknown"

            record = {
                "id": (row.get("id") or "").strip(),
                "source_text": src,
                "target_text": tgt,
                "source_url": (row.get("source_url") or "").strip(),
                "register": register,
                "dialect": dialect,
                "error_types": _split(row.get("error_types") or ""),
                "annotator_ids": _split(row.get("annotator_ids") or ""),
                "notes": (row.get("notes") or "").strip(),
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {written} records (dropped {dropped}, skipped {skipped} blank) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
