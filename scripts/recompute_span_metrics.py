"""Span-metrics recompute — span-aware headline metrics from released hypotheses.

Audit rows 3.1/3.3 ask for one position-aware GEC scorer to replace the
position-agnostic exact-tuple scorer, and for trivial source==target pairs to
be dropped from the denominator. Both campaign 1 (audit retrain,
results/campaign_1_audit_retrain) and campaign 2 (multiseed,
results/campaign_2_multiseed) store source/hypothesis/reference per test
sentence, so the span-aware numbers come straight from those files — no GPU.

For each campaign this script computes, per run and pooled over the three seeds:
  * word-level F0.5  (legacy, evaluate_corpus) — for the side-by-side
  * span-aware F0.5  (evaluate_corpus_span)    — the new headline
on the FULL test set and on the EDITED subset (source != reference).

It then runs a paired span-aware bootstrap (morphaware vs baseline) on the
edited subset, pooling seeds by concatenation, and recomputes agreement
accuracy on hypotheses with both the legacy sentence-pass denominator and the
applicable-checks denominator.

Output: results/campaigns_span_metrics.json
"""

import json
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.evaluation.f05_scorer import evaluate_corpus, evaluate_corpus_span
from src.evaluation.gleu_scorer import compute_gleu
from src.evaluation.agreements import evaluate_agreement_accuracy
from src.evaluation.bootstrap import paired_bootstrap_f05

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGNS = {
    "campaign_1_audit_retrain": ROOT / "results" / "campaign_1_audit_retrain",
    "campaign_2_multiseed": ROOT / "results" / "campaign_2_multiseed",
}
BASE_SEEDS = ["baseline_seed42", "baseline_seed123", "baseline_seed777"]
MORPH_SEEDS = ["morphaware_seed42", "morphaware_seed123", "morphaware_seed777"]


def load_run(campaign_dir: Path, run: str):
    srcs, hyps, refs = [], [], []
    with open(campaign_dir / run / "hypotheses.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            srcs.append(rec["source"])
            hyps.append(rec["hypothesis"])
            refs.append(rec["reference"])
    return srcs, hyps, refs


def edited_only(srcs, hyps, refs):
    e_s, e_h, e_r = [], [], []
    for s, h, r in zip(srcs, hyps, refs):
        if s == r:
            continue
        e_s.append(s)
        e_h.append(h)
        e_r.append(r)
    return e_s, e_h, e_r


def score_run(srcs, hyps, refs):
    word = evaluate_corpus(srcs, hyps, refs)
    span, span_by_type = evaluate_corpus_span(srcs, hyps, refs)
    gleu = compute_gleu(srcs, hyps, refs)
    agr = evaluate_agreement_accuracy(hyps)
    return {
        "n": len(srcs),
        "word_f05": word.f05,
        "word_p": word.precision,
        "word_r": word.recall,
        "word_tp": word.tp,
        "word_fp": word.fp,
        "word_fn": word.fn,
        "span_f05": span.f05,
        "span_p": span.precision,
        "span_r": span.recall,
        "span_tp": span.tp,
        "span_fp": span.fp,
        "span_fn": span.fn,
        "span_by_type": {
            t: {"f05": m.f05, "p": m.precision, "r": m.recall,
                "tp": m.tp, "fp": m.fp, "fn": m.fn}
            for t, m in span_by_type.items()
        },
        "gleu": gleu,
        "agr_accuracy_legacy": agr["accuracy"],
        "agr_accuracy_applicable": agr["accuracy_applicable"],
        "agr_applicable_sentences": agr["applicable_sentences"],
        "agr_avg_checks_applicable": agr["avg_checks_applicable"],
    }


def mean_std(values):
    if not values:
        return {"mean": 0.0, "std": 0.0}
    if len(values) == 1:
        return {"mean": values[0], "std": 0.0}
    return {"mean": statistics.mean(values), "std": statistics.stdev(values)}


def pool_arm(campaign_dir, seeds, subset):
    """Return concatenated (srcs, hyps, refs) and per-seed scores for an arm."""
    per_seed = {}
    cat_s, cat_h, cat_r = [], [], []
    for run in seeds:
        srcs, hyps, refs = load_run(campaign_dir, run)
        if subset == "edited":
            srcs, hyps, refs = edited_only(srcs, hyps, refs)
        per_seed[run] = score_run(srcs, hyps, refs)
        cat_s += srcs
        cat_h += hyps
        cat_r += refs
    return (cat_s, cat_h, cat_r), per_seed


def summarize_arm(per_seed):
    keys = ["word_f05", "span_f05", "gleu",
            "agr_accuracy_legacy", "agr_accuracy_applicable"]
    return {k: mean_std([per_seed[r][k] for r in per_seed]) for k in keys}


def main():
    out = {}
    for camp_name, camp_dir in CAMPAIGNS.items():
        camp = {}
        for subset in ("full", "edited"):
            (b_s, b_h, b_r), base_seeds = pool_arm(camp_dir, BASE_SEEDS, subset)
            (m_s, m_h, m_r), morph_seeds = pool_arm(camp_dir, MORPH_SEEDS, subset)

            sub = {
                "baseline": {
                    "per_seed": base_seeds,
                    "summary": summarize_arm(base_seeds),
                },
                "morphaware": {
                    "per_seed": morph_seeds,
                    "summary": summarize_arm(morph_seeds),
                },
            }

            # Paired bootstrap (morphaware vs baseline) on pooled seeds.
            # Both arms share the same source/reference order within a seed,
            # so concatenation keeps the pairing aligned.
            for label, scoring in (("span", "span"), ("word", "word")):
                bs = paired_bootstrap_f05(
                    sources=m_s,
                    hypotheses_a=m_h,
                    hypotheses_b=b_h,
                    references=m_r,
                    n_resamples=1000,
                    seed=42,
                    scoring=scoring,
                )
                sub[f"bootstrap_{label}"] = {
                    "f05_morphaware": bs.f05_a,
                    "f05_baseline": bs.f05_b,
                    "delta_morph_minus_base": bs.delta,
                    "ci_low": bs.ci_low,
                    "ci_high": bs.ci_high,
                    "p_value": bs.p_value,
                    "n_sentences": bs.n_sentences,
                    "n_resamples": bs.n_resamples,
                }
            camp[subset] = sub
        out[camp_name] = camp

    out_path = ROOT / "results" / "campaigns_span_metrics.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    # Console summary
    for camp_name, camp in out.items():
        print(f"\n=== {camp_name} ===")
        for subset in ("full", "edited"):
            b = camp[subset]["baseline"]["summary"]
            m = camp[subset]["morphaware"]["summary"]
            bspan = camp[subset]["bootstrap_span"]
            print(f"  [{subset}]")
            print(f"    baseline   word_F0.5={b['word_f05']['mean']:.4f}  "
                  f"span_F0.5={b['span_f05']['mean']:.4f}  "
                  f"agr_appl={b['agr_accuracy_applicable']['mean']:.4f}  "
                  f"agr_legacy={b['agr_accuracy_legacy']['mean']:.4f}")
            print(f"    morphaware word_F0.5={m['word_f05']['mean']:.4f}  "
                  f"span_F0.5={m['span_f05']['mean']:.4f}  "
                  f"agr_appl={m['agr_accuracy_applicable']['mean']:.4f}  "
                  f"agr_legacy={m['agr_accuracy_legacy']['mean']:.4f}")
            print(f"    bootstrap(span) delta={bspan['delta_morph_minus_base']:+.4f}  "
                  f"95% CI [{bspan['ci_low']:+.4f},{bspan['ci_high']:+.4f}]  "
                  f"p={bspan['p_value']:.4f}  n={bspan['n_sentences']}")
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
