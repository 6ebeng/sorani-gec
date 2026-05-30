"""Paired bootstrap significance tests across all GEC systems.

Reviewer item R9: report p-values and reframe overlapping confidence
intervals. This loads the per-sentence hypotheses dumped for the neural
checkpoints (results/phase_d/<run>/hypotheses.jsonl) and the non-neural
baselines (results/baselines/<system>_hypotheses.jsonl), then runs the
sentence-level paired bootstrap for the comparisons that matter:

  - ByT5 baseline (seed 42) vs morphology-aware (seed 42)   [the headline gap]
  - best n-gram LM vs ByT5 baseline                          [non-neural beats neural?]
  - reverse-rule vs ByT5 baseline

Writes results/baselines/bootstrap_pvalues.json.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.evaluation.bootstrap import paired_bootstrap_f05

PHASE_D = "results/phase_d"
BASELINES = "results/baselines"
N_RESAMPLES = 2000
SEED = 42


def load_hyp(path: str):
    srcs, hyps, refs = [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            srcs.append(rec["source"])
            hyps.append(rec["hypothesis"])
            refs.append(rec["reference"])
    return srcs, hyps, refs


def main():
    systems = {}

    # neural
    for run in ["baseline_seed42", "baseline_seed123", "baseline_seed777",
                "morphaware_seed42", "morphaware_seed123", "morphaware_seed777"]:
        p = f"{PHASE_D}/{run}/hypotheses.jsonl"
        if os.path.exists(p):
            systems[run] = load_hyp(p)

    # non-neural
    for sysname in ["copy", "hunspell", "reverse_rule", "ngram_lm"]:
        p = f"{BASELINES}/{sysname}_hypotheses.jsonl"
        if os.path.exists(p):
            systems[sysname] = load_hyp(p)

    print("Loaded systems:", ", ".join(systems.keys()))

    # sanity: shared sources/references
    ref_src, _, ref_ref = next(iter(systems.values()))

    comparisons = [
        ("baseline_seed42", "morphaware_seed42"),
        ("ngram_lm", "baseline_seed42"),
        ("reverse_rule", "baseline_seed42"),
        ("ngram_lm", "morphaware_seed42"),
        ("ngram_lm", "reverse_rule"),
        ("reverse_rule", "hunspell"),
    ]

    results = []
    for a, b in comparisons:
        if a not in systems or b not in systems:
            print(f"  skip {a} vs {b} (missing)")
            continue
        srcs_a, hyps_a, refs_a = systems[a]
        srcs_b, hyps_b, refs_b = systems[b]
        res = paired_bootstrap_f05(
            sources=srcs_a, hypotheses_a=hyps_a, hypotheses_b=hyps_b,
            references=refs_a, n_resamples=N_RESAMPLES, seed=SEED,
        )
        print(f"\n{a}  vs  {b}")
        print(f"  {res}")
        results.append({
            "system_a": a, "system_b": b,
            "f05_a": res.f05_a, "f05_b": res.f05_b,
            "delta": res.delta, "ci_low": res.ci_low, "ci_high": res.ci_high,
            "p_value": res.p_value, "n_resamples": res.n_resamples,
            "n_sentences": res.n_sentences,
        })

    # Pooled multi-seed baseline vs morphaware (L10-06): instead of resting the
    # headline gap on the seed-42 checkpoints alone, pool the matched per-seed
    # hypotheses across all three seeds (each test sentence contributes once per
    # seed) so seed variance enters the resampling distribution directly.
    pool_seeds = ["42", "123", "777"]
    have_all = all(
        f"baseline_seed{s}" in systems and f"morphaware_seed{s}" in systems
        for s in pool_seeds
    )
    if have_all:
        p_src, p_a, p_b, p_ref = [], [], [], []
        for s in pool_seeds:
            b_src, b_hyp, b_ref = systems[f"baseline_seed{s}"]
            _m_src, m_hyp, _m_ref = systems[f"morphaware_seed{s}"]
            p_src.extend(b_src)
            p_a.extend(b_hyp)
            p_b.extend(m_hyp)
            p_ref.extend(b_ref)
        res = paired_bootstrap_f05(
            sources=p_src, hypotheses_a=p_a, hypotheses_b=p_b,
            references=p_ref, n_resamples=N_RESAMPLES, seed=SEED,
        )
        print("\nbaseline_pooled3seed  vs  morphaware_pooled3seed")
        print(f"  {res}")
        results.append({
            "system_a": "baseline_pooled3seed", "system_b": "morphaware_pooled3seed",
            "f05_a": res.f05_a, "f05_b": res.f05_b,
            "delta": res.delta, "ci_low": res.ci_low, "ci_high": res.ci_high,
            "p_value": res.p_value, "n_resamples": res.n_resamples,
            "n_sentences": res.n_sentences,
            "pooled_seeds": pool_seeds,
        })
    else:
        print("  skip pooled 3-seed baseline vs morphaware (missing per-seed hypotheses)")

    with open(f"{BASELINES}/bootstrap_pvalues.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved -> {BASELINES}/bootstrap_pvalues.json")


if __name__ == "__main__":
    main()
