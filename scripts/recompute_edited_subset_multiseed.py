"""Recompute edited-subset metrics from the released multiseed-campaign hypotheses.

The released campaign-2 multiseed runs are the gated-residual variant. The
stored hypotheses.jsonl files already contain source/hypothesis/reference per
test sentence, so the edited-subset numbers can be recomputed without a GPU.
The metrics match the retired campaign evaluator: F0.5, GLEU, agreement
accuracy, and a CER-floor gate at 0.5.

Edited subset = pairs where source != reference.
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.evaluation.f05_scorer import evaluate_corpus
from src.evaluation.gleu_scorer import compute_gleu
from src.evaluation.agreement_accuracy import evaluate_agreement_accuracy

MULTISEED_RESULTS = Path("results/campaign_2_multiseed")
RUNS = [
    "baseline_seed42", "baseline_seed123", "baseline_seed777",
    "morphaware_seed42", "morphaware_seed123", "morphaware_seed777",
]


def levenshtein(a: str, b: str) -> int:
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def cer_vs_ref(hyps, refs) -> float:
    tot_d = tot_l = 0
    for h, r in zip(hyps, refs):
        tot_d += levenshtein(h, r)
        tot_l += max(len(r), 1)
    return tot_d / tot_l if tot_l else 0.0


def avg_cer_vs_src(srcs, hyps) -> float:
    # Matches phase3_evaluate_all gate: editdistance(src, hyp)/len(src), averaged.
    return sum(
        levenshtein(s, h) / max(len(s), 1) for s, h in zip(srcs, hyps)
    ) / max(len(srcs), 1)


def load_edited(run: str):
    srcs, hyps, refs = [], [], []
    with open(MULTISEED_RESULTS / run / "hypotheses.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            s, h, r = rec["source"], rec["hypothesis"], rec["reference"]
            if s == r:  # trivial pair, drop for edited subset
                continue
            srcs.append(s)
            hyps.append(h)
            refs.append(r)
    return srcs, hyps, refs


def evaluate(run: str):
    srcs, hyps, refs = load_edited(run)
    m = evaluate_corpus(srcs, hyps, refs)
    gleu = compute_gleu(srcs, hyps, refs)
    agr = evaluate_agreement_accuracy(hyps)["accuracy"]
    cer_ref = cer_vs_ref(hyps, refs)
    cer_src = avg_cer_vs_src(srcs, hyps)
    floored = 0.0 if cer_src > 0.5 else agr
    return {
        "run": run, "n": len(srcs), "f05": m.f05, "p": m.precision,
        "r": m.recall, "tp": m.tp, "fp": m.fp, "fn": m.fn,
        "gleu": gleu, "cer_ref": cer_ref, "cer_src": cer_src,
        "agr_raw": agr, "agr_floor": floored,
    }


def main():
    rows = [evaluate(run) for run in RUNS]
    hdr = (f"{'run':<20}{'n':>5}{'F0.5':>9}{'GLEU':>9}{'CER_ref':>9}"
           f"{'CER_src':>9}{'Agr_raw':>9}{'Agr_flr':>9}")
    print(hdr)
    print("-" * len(hdr))
    for x in rows:
        print(f"{x['run']:<20}{x['n']:>5}{x['f05']:>9.4f}{x['gleu']:>9.4f}"
              f"{x['cer_ref']:>9.4f}{x['cer_src']:>9.4f}{x['agr_raw']:>9.4f}"
              f"{x['agr_floor']:>9.4f}")

    def mean(keys, field):
        vals = [x[field] for x in rows if x["run"] in keys]
        return sum(vals) / len(vals)

    base = ["baseline_seed42", "baseline_seed123", "baseline_seed777"]
    morph = ["morphaware_seed42", "morphaware_seed123", "morphaware_seed777"]
    print("\nMEANS (edited subset):")
    for label, keys in [("baseline", base), ("morphaware", morph)]:
        print(f"  {label:<11} F0.5={mean(keys,'f05'):.4f}  GLEU={mean(keys,'gleu'):.4f}  "
              f"CER_ref={mean(keys,'cer_ref'):.4f}  CER_src={mean(keys,'cer_src'):.4f}  "
              f"Agr_raw={mean(keys,'agr_raw'):.4f}  Agr_flr={mean(keys,'agr_floor'):.4f}")

    out = MULTISEED_RESULTS / "edited_subset_recomputed.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
