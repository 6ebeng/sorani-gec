"""Export a CSV review template for the natural test set.

Seeds the CSV with the 287 active (source != target) pairs from
`data/splits/test.jsonl` so the format is concrete. The user reviews
each row, edits/replaces sentences with real organic-error examples,
and runs `csv_to_natural_jsonl.py` to produce
`data/natural_test/sentences.jsonl`.

Columns mirror the JSONL schema in data/natural_test/README.md.

Usage (PowerShell):
    python scripts/export_natural_csv.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST = ROOT / "data" / "splits" / "test.jsonl"
OUT = ROOT / "data" / "natural_test" / "sentences_for_review.csv"

COLUMNS = [
    "id",
    "source_text",
    "target_text",
    "error_types",       # semicolon-joined; e.g. "subject_verb;orthography"
    "source_url",
    "register",          # news | social | blog | learner | chat | forum | other
    "dialect",           # central | northern | southern | mixed | unknown
    "annotator_ids",     # semicolon-joined; e.g. "A1;A2"
    "notes",
    "keep",              # y / n / edit  -- review column
]


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    seen = 0
    with TEST.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            src = rec.get("source", "")
            tgt = rec.get("target", "")
            if not src or not tgt or src == tgt:
                continue
            seen += 1
            rid = f"nat_seed_{seen:04d}"
            etypes = ";".join(
                e.get("type", "") for e in (rec.get("errors") or []) if isinstance(e, dict)
            )
            rows.append({
                "id": rid,
                "source_text": src,
                "target_text": tgt,
                "error_types": etypes,
                "source_url": f"synthetic_seed:{rec.get('source_id', '?')}",
                "register": "other",
                "dialect": "central",
                "annotator_ids": "A1",
                "notes": f"category={rec.get('category', '?')}",
                "keep": "",
            })

    # UTF-8 with BOM so Excel opens Kurdish text correctly.
    with OUT.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} review rows to {OUT}")
    print("Columns:", ", ".join(COLUMNS))
    print("Open in Excel / VS Code, edit rows, mark `keep` = y/n/edit, then run:")
    print("    python scripts/csv_to_natural_jsonl.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
