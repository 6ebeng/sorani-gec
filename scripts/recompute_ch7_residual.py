"""Recompute the Chapter 7 detail tables from the released residual hypotheses.

Produces, for the seed-42 representative checkpoints (the convention the
significance section already uses):
  - Full-set agreement per-check + overall (raw and CER-floored)
  - Edited-subset per-error-type F0.5 (baseline vs morphaware)
  - Edited-subset span-based TP

All numbers come from the stored hypotheses.jsonl (gated-residual runs);
no GPU is needed. Order in hypotheses.jsonl matches data/splits_v2/test.jsonl.
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.evaluation.f05_scorer import evaluate_corpus, evaluate_corpus_span
from src.evaluation.agreement_accuracy import (
    evaluate_agreement_accuracy,
    evaluate_agreement_by_check,
)

MULTISEED_RESULTS = Path("results/campaign_2_multiseed")
TEST = Path("data/splits_v2/test.jsonl")


def levenshtein(a, b):
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if a[i - 1] == b[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def load(run):
    rows = []
    with open(MULTISEED_RESULTS / run / "hypotheses.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def primary_types():
    types = []
    with open(TEST, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            errs = rec.get("errors", [])
            types.append(errs[0]["type"] if errs else None)
    return types


def full_set_agreement(run):
    rows = load(run)
    hyps = [r["hypothesis"] for r in rows]
    overall = evaluate_agreement_accuracy(hyps)
    by_check = evaluate_agreement_by_check(hyps)
    # CER-floor (vs source) gate on full set
    avg_cer = sum(
        levenshtein(r["source"], r["hypothesis"]) / max(len(r["source"]), 1)
        for r in rows
    ) / max(len(rows), 1)
    floored = 0.0 if avg_cer > 0.5 else overall["accuracy"]
    return overall, by_check, avg_cer, floored


def edited_per_type(run, types):
    rows = load(run)
    assert len(rows) == len(types), f"{len(rows)} != {len(types)}"
    buckets = defaultdict(lambda: {"s": [], "h": [], "r": []})
    for rec, t in zip(rows, types):
        if rec["source"] == rec["reference"]:
            continue  # edited subset only
        buckets[t]["s"].append(rec["source"])
        buckets[t]["h"].append(rec["hypothesis"])
        buckets[t]["r"].append(rec["reference"])
    out = {}
    for t, b in buckets.items():
        m = evaluate_corpus(b["s"], b["h"], b["r"])
        out[t] = {"n": len(b["s"]), "f05": m.f05}
    return out


def edited_span_tp(run):
    rows = [r for r in load(run) if r["source"] != r["reference"]]
    s = [r["source"] for r in rows]
    h = [r["hypothesis"] for r in rows]
    ref = [r["reference"] for r in rows]
    overall, _ = evaluate_corpus_span(s, h, ref)
    return overall.tp, overall.fp, overall.fn


def main():
    types = primary_types()
    print(f"Loaded {len(types)} test primary-types\n")

    print("=== FULL-SET AGREEMENT (seed42 representative) ===")
    for run in ["baseline_seed42", "morphaware_seed42"]:
        overall, by_check, avg_cer, floored = full_set_agreement(run)
        print(f"\n{run}: overall_raw={overall['accuracy']:.4f}  "
              f"avg_cer_vs_src={avg_cer:.4f}  overall_floored={floored:.4f}")
        for label, info in by_check["per_check"].items():
            print(f"   {label:<22} {info['accuracy']:.4f}")

    print("\n=== EDITED-SUBSET PER-TYPE F0.5 (seed42) ===")
    bt = edited_per_type("baseline_seed42", types)
    mt = edited_per_type("morphaware_seed42", types)
    allt = sorted(set(bt) | set(mt), key=lambda t: -(bt.get(t, {}).get("n", 0)))
    print(f"{'type':<26}{'n':>5}{'base_F05':>10}{'morph_F05':>11}")
    for t in allt:
        n = bt.get(t, mt.get(t, {})).get("n", 0)
        print(f"{str(t):<26}{n:>5}{bt.get(t,{}).get('f05',0):>10.4f}"
              f"{mt.get(t,{}).get('f05',0):>11.4f}")

    print("\n=== EDITED-SUBSET SPAN TP (seed42) ===")
    for run in ["baseline_seed42", "morphaware_seed42"]:
        tp, fp, fn = edited_span_tp(run)
        print(f"  {run}: TP={tp}  FP={fp}  FN={fn}")


if __name__ == "__main__":
    main()
