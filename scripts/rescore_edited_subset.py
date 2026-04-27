"""Rescore baseline and morphology-aware predictions on the edited-only test
subset.

Joins existing model predictions in
``results/metrics_remote/{baseline,morphaware}/`` against the gold
``data/splits/test.jsonl`` (which carries the true target and per-error type
tags), filters to pairs where ``source != target``, and recomputes:

  * F0.5 (overall + per-error-type)
  * GLEU
  * CER (charcter error rate)
  * Agreement accuracy (raw and CER-floored at <0.5)

This addresses the data-pipeline diagnosis: the original test set contained
~71% trivial copy-pairs that drowned the metric. The edited subset (n=287)
is the actual GEC test set.

Outputs JSON to ``results/metrics_remote/<name>/rescored_edited.json``.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

from src.evaluation import (
    AgreementChecker,
    compute_gleu,
    evaluate_corpus,
    evaluate_corpus_by_type,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "data" / "splits"
RESULTS = ROOT / "results" / "metrics_remote"


def char_error_rate(hyp: str, ref: str) -> float:
    """Levenshtein-distance / |ref|, computed at codepoint level."""
    if not ref:
        return 0.0 if not hyp else 1.0
    n, m = len(hyp), len(ref)
    if n == 0:
        return 1.0
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        curr = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if hyp[i - 1] == ref[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[m] / m


def load_gold(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]


def load_preds(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]


def rescore(name: str) -> dict:
    pred_dir = RESULTS / name
    preds = load_preds(pred_dir / "evaluation_pairs.jsonl")
    gold = load_gold(SPLITS / "test.jsonl")

    # Join: gold has source/target/errors; preds have source/corrected.
    # Sources may be normalised differently — match by index instead, since
    # eval was run against the same test set in the same order.
    assert len(preds) == len(gold), f"{len(preds)} preds vs {len(gold)} gold"

    # Build joined records, keeping only edited pairs (source != target).
    edited = []
    for p, g in zip(preds, gold):
        if g["source"] == g["target"]:
            continue
        # Use gold source, not pred source: gold may have whitespace
        # the inference loop dropped.
        edited.append({
            "source": g["source"],
            "target": g["target"],
            "hyp": p.get("corrected", ""),
            "errors": g.get("errors", []),
        })
    log.info("[%s] joined %d preds + %d gold -> %d edited pairs", name, len(preds), len(gold), len(edited))

    sources = [r["source"] for r in edited]
    targets = [r["target"] for r in edited]
    hyps = [r["hyp"] for r in edited]

    # Overall F0.5
    f05_metrics = evaluate_corpus(sources=sources, hypotheses=hyps, references=targets)

    # Per-error-type F0.5: assign each pair its primary (first) error type
    primary_types = [r["errors"][0]["type"] if r["errors"] else "no_error" for r in edited]
    by_type = evaluate_corpus_by_type(
        sources=sources,
        hypotheses=hyps,
        references=targets,
        error_types=primary_types,
    )

    # GLEU
    gleu = compute_gleu(hypotheses=hyps, references=targets, sources=sources)

    # CER per pair
    cers = [char_error_rate(h, t) for h, t in zip(hyps, targets)]
    mean_cer = sum(cers) / len(cers)

    # Agreement accuracy with CER floor
    checker = AgreementChecker()
    raw_pass = 0
    floored_pass = 0
    for h, c in zip(hyps, cers):
        result = checker.check_sentence(h)
        if result.is_correct:
            raw_pass += 1
            if c < 0.5:
                floored_pass += 1

    n = len(edited)
    out = {
        "model": name,
        "n_edited": n,
        "n_skipped_trivial_copies": len(preds) - n,
        "f05": {
            "precision": f05_metrics.precision,
            "recall": f05_metrics.recall,
            "f05": f05_metrics.f05,
            "tp": f05_metrics.tp,
            "fp": f05_metrics.fp,
            "fn": f05_metrics.fn,
        },
        "gleu": gleu,
        "cer_mean": mean_cer,
        "agreement_raw": raw_pass / n,
        "agreement_cer_floor_0.5": floored_pass / n,
        "f05_per_type": {
            t: {
                "precision": m.precision,
                "recall": m.recall,
                "f05": m.f05,
                "n_pairs": primary_types.count(t),
                "tp": m.tp, "fp": m.fp, "fn": m.fn,
            }
            for t, m in by_type.items()
        },
        "type_distribution": dict(Counter(primary_types).most_common()),
    }
    out_path = pred_dir / "rescored_edited.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("[%s] wrote %s", name, out_path)
    return out


def main() -> None:
    for name in ("baseline", "morphaware"):
        out = rescore(name)
        log.info(
            "[%s] n=%d  F0.5=%.4f  GLEU=%.4f  CER=%.3f  AgrAcc(raw)=%.3f  AgrAcc(floor)=%.3f",
            name,
            out["n_edited"],
            out["f05"]["f05"],
            out["gleu"],
            out["cer_mean"],
            out["agreement_raw"],
            out["agreement_cer_floor_0.5"],
        )


if __name__ == "__main__":
    main()
