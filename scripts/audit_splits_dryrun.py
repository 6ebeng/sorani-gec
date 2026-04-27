"""Dry-run audit: pipe existing splits through the patched normalizer +
strict-Sorani gate and report how many records would survive.

Does NOT modify any files. Run from repo root:
    python scripts/audit_splits_dryrun.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.normalizer import SoraniNormalizer  # noqa: E402

# Load 02_normalize.py as a module to reuse passes_quality_gate
spec = importlib.util.spec_from_file_location(
    "_norm_script", ROOT / "scripts" / "02_normalize.py"
)
norm_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(norm_mod)

SPLITS = ROOT / "data" / "splits"

# Detection regexes
ARABIC_ONLY_RE = re.compile(r"[\u0629\u064B-\u0652\u0670]")  # TEH MARBUTA + harakat
SORANI_DISTINCT_RE = re.compile(
    r"[\u0695\u06B5\u06CE\u06C6\u06D5\u06A4\u0698\u06AF\u067E\u0686\u06A9]"
)


def audit(path: Path) -> dict:
    norm = SoraniNormalizer(remove_diacritics=True, normalize_chars=True)
    total = 0
    arabic_pre = 0
    arabic_post = 0
    no_sorani_distinct = 0
    gate_fails = {}
    examples_arabic_post = []
    examples_no_sorani = []

    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            rec = json.loads(line)
            for field in ("source", "target"):
                text = rec.get(field, "")
                if not text:
                    continue
                if ARABIC_ONLY_RE.search(text):
                    arabic_pre += 1
                normed = norm.normalize(text)
                if ARABIC_ONLY_RE.search(normed):
                    arabic_post += 1
                    if len(examples_arabic_post) < 5:
                        examples_arabic_post.append((rec.get("id", "?"), field, normed[:80]))
                if not SORANI_DISTINCT_RE.search(normed):
                    no_sorani_distinct += 1
                    if len(examples_no_sorani) < 5:
                        examples_no_sorani.append((rec.get("id", "?"), field, normed[:80]))
                passes, reason = norm_mod.passes_quality_gate(normed, strict_sorani=True)
                if not passes:
                    gate_fails[reason] = gate_fails.get(reason, 0) + 1

    return {
        "file": path.name,
        "records": total,
        "arabic_chars_pre_normalize": arabic_pre,
        "arabic_chars_post_normalize": arabic_post,
        "no_sorani_distinctive_letter": no_sorani_distinct,
        "strict_gate_fail_reasons": gate_fails,
        "examples_arabic_post": examples_arabic_post,
        "examples_no_sorani": examples_no_sorani,
    }


def main() -> int:
    results = []
    for name in ("train.jsonl", "dev.jsonl", "test.jsonl"):
        p = SPLITS / name
        if not p.exists():
            print(f"SKIP {p} (not found)")
            continue
        results.append(audit(p))

    print("\n=== Dry-run audit (patched normalizer) ===\n")
    for r in results:
        print(f"## {r['file']}  ({r['records']} records)")
        print(f"  Arabic-only chars present BEFORE normalize : {r['arabic_chars_pre_normalize']}")
        print(f"  Arabic-only chars present AFTER  normalize : {r['arabic_chars_post_normalize']}")
        print(f"  Records with NO Sorani-distinctive letter  : {r['no_sorani_distinctive_letter']}")
        print(f"  Strict-gate failure reasons                 : {r['strict_gate_fail_reasons']}")
        if r["examples_arabic_post"]:
            print("  Sample Arabic-leak survivors:")
            for ex in r["examples_arabic_post"]:
                print(f"    {ex[0]} [{ex[1]}] {ex[2]!r}")
        if r["examples_no_sorani"]:
            print("  Sample non-Sorani lines:")
            for ex in r["examples_no_sorani"]:
                print(f"    {ex[0]} [{ex[1]}] {ex[2]!r}")
        print()

    out = SPLITS.parent.parent / "results" / "data_diagnosis" / "splits_dryrun_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
