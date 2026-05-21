"""Replace stray Latin punctuation in split `source` fields with the Arabic
equivalent and prune the matching entries from each record's `errors` list.

Background: an earlier version of `PunctuationErrorGenerator` could inject
Latin `,` `;` `?` as error targets. The thesis evaluates agreement /
morphological GEC, so Latin↔Arabic punctuation swaps are out of scope and
were removed from the generator. This script repairs already-written splits
without re-running the full pipeline.
"""

from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path

LATIN_TO_ARABIC = {",": "،", ";": "؛", "?": "؟"}
SPLITS_DIR = Path(__file__).resolve().parents[1] / "data" / "splits"


def clean_record(rec: dict) -> tuple[dict, bool]:
    source = rec.get("source", "")
    if not any(ch in source for ch in LATIN_TO_ARABIC):
        return rec, False

    new_source = source
    for latin, arabic in LATIN_TO_ARABIC.items():
        new_source = new_source.replace(latin, arabic)
    rec["source"] = new_source

    cleaned_errors = []
    for err in rec.get("errors", []):
        injected = err.get("error", "")
        if err.get("type") == "punctuation" and injected in LATIN_TO_ARABIC:
            continue
        cleaned_errors.append(err)
    rec["errors"] = cleaned_errors
    return rec, True


def process(path: Path) -> Counter:
    stats = Counter()
    backup = path.with_suffix(path.suffix + ".pre_latin_clean.bak")
    if not backup.exists():
        shutil.copy2(path, backup)

    with backup.open(encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh]

    out_lines = []
    for rec in records:
        new_rec, changed = clean_record(rec)
        if changed:
            stats["records_changed"] += 1
            if new_rec["source"] == new_rec["target"]:
                stats["became_identity"] += 1
        out_lines.append(json.dumps(new_rec, ensure_ascii=False))

    path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    stats["total"] = len(records)
    return stats


def main() -> None:
    for name in ("train.jsonl", "dev.jsonl", "test.jsonl"):
        path = SPLITS_DIR / name
        if not path.exists():
            print(f"skip {name}: not found")
            continue
        stats = process(path)
        print(f"{name}: total={stats['total']} changed={stats['records_changed']} now_identity={stats['became_identity']}")


if __name__ == "__main__":
    main()
