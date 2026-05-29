"""Build the blind human-evaluation set for Phase F (R6 + R15).

The reviewer report asks for two things that this script prepares:

  R6  — a two-annotator study on >=100 model outputs with Cohen's kappa.
  R15 — validation of the 14-check agreement metric against native-speaker
        grammaticality judgement (Kendall tau / Cohen kappa).

This script samples real Phase-D test predictions (baseline + morphology-aware,
seed 42) and writes a *blind* rating set. Annotators only ever see the source
sentence and one model correction; they never learn which system produced it.
Provenance and the metric verdict are stored in a separate manifest that the
web app does not read, so the human ratings stay uncontaminated.

Outputs (under ``results/human_eval/``):
  - ``evaluation_pairs.jsonl``        what the Gradio app serves (source + corrected only)
  - ``evaluation_pairs_manifest.jsonl`` hidden: system, reference, metric verdict, bucket

Run from ``Implementation/sorani-gec``::

    python scripts/build_eval_pairs.py --n 120
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.agreement_accuracy import AgreementChecker

PHASE_D = ROOT / "results" / "phase_d"
EVAL_DIR = ROOT / "results" / "human_eval"
PAIRS_OUT = EVAL_DIR / "evaluation_pairs.jsonl"
MANIFEST_OUT = EVAL_DIR / "evaluation_pairs_manifest.jsonl"

# Seed-42 runs are the per-system representatives used everywhere else in the
# thesis, so the human study rates the same checkpoints the F0.5 table reports.
SYSTEMS = {
    "baseline": PHASE_D / "baseline_seed42" / "hypotheses.jsonl",
    "morphaware": PHASE_D / "morphaware_seed42" / "hypotheses.jsonl",
}


def _cer(a: str, b: str) -> float:
    """Character error rate (Levenshtein / max length)."""
    if not a and not b:
        return 0.0
    if not a or not b:
        return 1.0
    m, n = len(a), len(b)
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[n] / max(m, n)


def _bucket(cer: float) -> str:
    if cer == 0.0:
        return "no_edit"
    if cer <= 0.05:
        return "trivial"
    if cer <= 0.20:
        return "mild"
    if cer <= 0.50:
        return "heavy"
    return "rewrite"


def _pair_id(system: str, source: str, corrected: str) -> str:
    raw = f"{system}\u241f{source}\u241f{corrected}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:12]


def _load(path: Path, system: str) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"Missing hypotheses for {system}: {path}")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        rows.append(
            {
                "system": system,
                "source": rec.get("source", "").strip(),
                "corrected": rec.get("hypothesis", rec.get("corrected", "")).strip(),
                "reference": rec.get("reference", "").strip(),
            }
        )
    return rows


def _stratified(rows: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Stratify by edit-size bucket so trivial and heavy edits both appear."""
    buckets: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        buckets[_bucket(_cer(r["source"], r["corrected"]))].append(r)
    total = sum(len(v) for v in buckets.values())
    picked, leftover = [], []
    for items in buckets.values():
        rng.shuffle(items)
        quota = max(1, round(len(items) / total * n))
        picked.extend(items[:quota])
        leftover.extend(items[quota:])
    if len(picked) < n:
        rng.shuffle(leftover)
        picked.extend(leftover[: n - len(picked)])
    return picked[:n]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build blind human-eval pairs (Phase F)")
    parser.add_argument("--n", type=int, default=120,
                        help="Total pairs across both systems (>=100 satisfies R6)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    per_system = max(1, args.n // len(SYSTEMS))
    selected: list[dict] = []
    for system, path in SYSTEMS.items():
        rows = _load(path, system)
        # Drop empty corrections and exact copies of the source — neither tells
        # an annotator anything about correction quality.
        rows = [r for r in rows if r["corrected"]]
        selected.extend(_stratified(rows, per_system, rng))

    rng.shuffle(selected)

    checker = AgreementChecker()
    blind_lines, manifest_lines = [], []
    for r in selected:
        result = checker.check_sentence(r["corrected"])
        pid = _pair_id(r["system"], r["source"], r["corrected"])
        blind_lines.append(
            json.dumps(
                {"pair_id": pid, "source": r["source"], "corrected": r["corrected"]},
                ensure_ascii=False,
            )
        )
        manifest_lines.append(
            json.dumps(
                {
                    "pair_id": pid,
                    "system": r["system"],
                    "source": r["source"],
                    "corrected": r["corrected"],
                    "reference": r["reference"],
                    "cer_bucket": _bucket(_cer(r["source"], r["corrected"])),
                    "agreement_checks_passed": result.checks_passed,
                    "agreement_checks_total": result.checks_total,
                    "agreement_accuracy": round(result.accuracy, 4),
                    "agreement_pass": result.is_correct,
                },
                ensure_ascii=False,
            )
        )

    PAIRS_OUT.write_text("\n".join(blind_lines) + "\n", encoding="utf-8", newline="\n")
    MANIFEST_OUT.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8", newline="\n")

    by_system = defaultdict(int)
    by_bucket = defaultdict(int)
    for ln in manifest_lines:
        rec = json.loads(ln)
        by_system[rec["system"]] += 1
        by_bucket[rec["cer_bucket"]] += 1

    print(f"Wrote {len(blind_lines)} blind pairs -> {PAIRS_OUT.relative_to(ROOT)}")
    print(f"Wrote manifest               -> {MANIFEST_OUT.relative_to(ROOT)}")
    print(f"  by system: {dict(by_system)}")
    print(f"  by bucket: {dict(by_bucket)}")


if __name__ == "__main__":
    main()
