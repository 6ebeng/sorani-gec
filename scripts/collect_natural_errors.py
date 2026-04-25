"""Scaffold for collecting naturally-occurring Sorani grammatical errors.

This script is an append-only recorder: it reads candidate sentences from
stdin (or a file), normalises them, and writes one JSON line per item to
`data/natural_test/sentences.jsonl`. It does NOT auto-correct, since the
whole point of this dataset is that corrections are human-written.

Usage:
    # Interactive — paste sentences, blank line to submit, Ctrl-D to finish
    python scripts/collect_natural_errors.py --annotator A1 --source learner_essay

    # Batch from file
    python scripts/collect_natural_errors.py --annotator A1 --source twitter \
        --input raw_tweets.txt --dialect central

Notes on workflow:
    1. First pass: collect source sentences only (leave `target_text` empty).
    2. Second pass: open the file and fill in `target_text` by hand.
    3. Third pass: `--review` mode lets a second annotator tick/adjust each entry.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import uuid
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.normalizer import normalize_sorani  # type: ignore  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "natural_test"
OUT_FILE = OUT_DIR / "sentences.jsonl"

VALID_REGISTERS = {"news", "social", "blog", "learner", "chat", "forum", "other"}
VALID_DIALECTS = {"central", "northern", "southern", "mixed", "unknown"}


def _next_id() -> str:
    return f"nat_{uuid.uuid4().hex[:8]}"


def _load_existing_ids() -> set[str]:
    if not OUT_FILE.exists():
        return set()
    seen: set[str] = set()
    with OUT_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                seen.add(json.loads(line)["id"])
            except (KeyError, json.JSONDecodeError):
                continue
    return seen


def _read_sentences(path: Path | None) -> list[str]:
    if path is None:
        print("Paste sentences, one per line. Ctrl-Z + Enter (Windows) or Ctrl-D (Unix) to finish.")
        return [line.strip() for line in sys.stdin.readlines() if line.strip()]
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect organic Sorani errors.")
    parser.add_argument("--annotator", required=True, help="Annotator ID (e.g. A1, A2).")
    parser.add_argument("--source", required=True, help="Provenance tag or URL.")
    parser.add_argument("--register", default="other", choices=sorted(VALID_REGISTERS))
    parser.add_argument("--dialect", default="unknown", choices=sorted(VALID_DIALECTS))
    parser.add_argument("--input", type=Path, default=None, help="Optional input file; else stdin.")
    parser.add_argument("--no-normalize", action="store_true", help="Skip normalisation (rare).")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    existing = _load_existing_ids()
    logger.info("Existing entries: %d", len(existing))

    sentences = _read_sentences(args.input)
    if not sentences:
        logger.warning("No input sentences.")
        return 0

    written = 0
    with OUT_FILE.open("a", encoding="utf-8") as out:
        for raw in sentences:
            text = raw if args.no_normalize else normalize_sorani(raw)
            record = {
                "id": _next_id(),
                "source_text": text,
                "target_text": "",
                "source_url": args.source,
                "register": args.register,
                "dialect": args.dialect,
                "error_types": [],
                "annotator_ids": [args.annotator],
                "notes": "",
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    logger.info("Wrote %d records to %s", written, OUT_FILE)
    logger.info("Next: open the file and fill in target_text + error_types.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
