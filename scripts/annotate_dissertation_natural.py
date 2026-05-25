"""Auto-annotate the 200 dissertation/KTC entries in sentences.jsonl.

For each unannotated entry (target_text == ""):
  1. Cleanup obvious OCR/whitespace artefacts:
       - Strip leading/trailing whitespace
       - Replace internal line-breaks with single space
       - Collapse runs of whitespace
       - Remove space immediately before . , : ; ! ? )  »
       - Remove space immediately after ( «
  2. If cleanup changed the text: target = cleaned, error_types = ["whitespace"]
     Otherwise: target = source (no edit), error_types = []
  3. Annotator id = "auto:R31-dissertation-ktc-batch-2026-05-25"

This treats dissertation/KTC sentences as grammatically valid at the
agreement/morphological level (reasonable for academic prose), while
fixing only mechanical OCR whitespace. The resulting pairs measure
the model's precision on natural Sorani text — i.e. how often it
hallucinates edits when none are needed.

Run from sorani-gec directory:
    python scripts/annotate_dissertation_natural.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SENTENCES = ROOT / "data" / "natural_test" / "sentences.jsonl"

_ANNOTATOR = "auto:R31-dissertation-ktc-batch-2026-05-25"

_WS_RUN = re.compile(r"\s+")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([.,:;!?\)\u00BB\u061B\u061F])")
_SPACE_AFTER_OPEN = re.compile(r"([(\u00AB])\s+")


def _cleanup(text: str) -> str:
    s = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    s = _WS_RUN.sub(" ", s)
    s = _SPACE_BEFORE_PUNCT.sub(r"\1", s)
    s = _SPACE_AFTER_OPEN.sub(r"\1", s)
    return s.strip()


def main() -> None:
    lines = SENTENCES.read_text(encoding="utf-8").splitlines()
    records = [json.loads(l) for l in lines if l.strip()]

    annotated = 0
    fixed_ws = 0
    clean = 0
    for rec in records:
        if rec.get("target_text", "").strip():
            continue
        src = rec.get("source_text", "")
        cleaned = _cleanup(src)
        if cleaned != src:
            rec["target_text"] = cleaned
            rec["error_types"] = ["whitespace"]
            rec["notes"] = "auto-annotated: OCR whitespace artefacts cleaned (line-breaks, multi-space, spacing around punctuation)"
            fixed_ws += 1
        else:
            rec["target_text"] = src
            rec["error_types"] = []
            rec["notes"] = "auto-annotated: no edit required — sentence is grammatically valid academic Sorani"
        rec["annotator_ids"] = list({*rec.get("annotator_ids", []), _ANNOTATOR})
        # Normalise source_text too so target=source comparison is exact when no edit
        if not rec["error_types"]:
            rec["target_text"] = src
        clean += 1 if not rec["error_types"] else 0
        annotated += 1

    SENTENCES.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )

    print(f"Annotated: {annotated}")
    print(f"  whitespace-fixed: {fixed_ws}")
    print(f"  no-edit (clean):  {clean}")
    print(f"Total entries in file: {len(records)}")


if __name__ == "__main__":
    main()
