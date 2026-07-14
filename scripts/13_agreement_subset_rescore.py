#!/usr/bin/env python
"""Re-score existing multiseed-campaign checkpoints on an agreement-only test subset.

No training, no GPU, no API. This reads the already-decoded hypotheses
(``results/campaign_2_multiseed/<run>/hypotheses.jsonl``) and the test split with its
per-pair error-type labels (``data/splits_v2/test.jsonl``), then asks a
narrower question than the headline tables: on the test pairs whose injected
error is a *structural agreement* phenomenon, does the morphology-aware model
differ from the baseline?

Why this is a fair question to ask post hoc. The morphological pathway was
designed to act on agreement (subject-verb, clitic, ezafe/noun-adjective,
case-role, tense, quantifier, ...), not on surface noise (orthography,
spelling, whitespace, punctuation). The full test set is 77% surface noise,
so a global metric dilutes any agreement-specific signal. Restricting to the
agreement subset is a scoping of the *evaluation*, not a change to the models
or the data, and it reuses the identical span-level paired bootstrap that
produced the headline significance numbers.

The script:
  1. Sanity-checks that re-scoring the FULL set reproduces the published
     per-seed F0.5 in ``eval_test.json`` (guards against row-misalignment).
  2. Reports per-seed F0.5 for baseline and morphaware on the agreement
     subset (edited pairs only).
  3. Runs the pooled (3-seed) span-level paired bootstrap on the subset.

Usage (from sorani-gec/):
    python scripts/13_agreement_subset_rescore.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Make ``src`` importable when run from the repo root.
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from src.evaluation.bootstrap import paired_bootstrap_f05  # noqa: E402
from src.evaluation.f05_scorer import evaluate_corpus  # noqa: E402

# --- configuration ---------------------------------------------------------

SEEDS = (42, 123, 777)
MULTISEED_RESULTS = _REPO / "results" / "campaign_2_multiseed"
TEST_JSONL = _REPO / "data" / "splits_v2" / "test.jsonl"

# Surface-noise generators the morphological pathway was never meant to fix.
SURFACE_TYPES = {
    "orthography",
    "spelling_confusion",
    "whitespace",
    "punctuation",
    "whitespace_error",
    "orthography_error",
}

# Tight "core agreement" set: the phenomena the agreement graph encodes
# directly. Used for the narrowest, most defensible claim.
CORE_AGREEMENT_TYPES = {
    "subject_verb_number",
    "subject_verb",
    "clitic_form",
    "clitic",
    "possessive_clitic",
    "noun_adjective_agreement",
    "case_role_preposition",
    "tense_agreement",
    "quantifier_agreement",
    "cross_clause_agreement",
    "conditional_agreement",
    "negative_concord",
    "vocative_imperative",
    "ergative",
}

OUT_JSON = _REPO / "results" / "campaign_2_multiseed" / "agreement_subset_rescore.json"


# --- loading ---------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def primary_error_type(row: dict) -> str | None:
    """The generator type of a test pair's first injected error, or None."""
    errors = row.get("errors") or []
    if not errors:
        return None
    return errors[0].get("type")


def load_run(run: str) -> list[dict]:
    path = MULTISEED_RESULTS / run / "hypotheses.jsonl"
    return load_jsonl(path)


# --- subset construction ---------------------------------------------------

def build_index_sets(test_rows: list[dict]) -> dict[str, list[int]]:
    """Row indices for each evaluation view.

    - ``edited``: any pair with source != target (the chapter's 397-pair set).
    - ``non_surface``: edited pairs whose primary type is not surface noise.
    - ``core_agreement``: edited pairs whose primary type is in the tight set.
    """
    edited, non_surface, core = [], [], []
    for i, row in enumerate(test_rows):
        src = row.get("source", "")
        tgt = row.get("target", "")
        if src == tgt:
            continue
        edited.append(i)
        etype = primary_error_type(row)
        if etype is None:
            continue
        if etype not in SURFACE_TYPES:
            non_surface.append(i)
        if etype in CORE_AGREEMENT_TYPES:
            core.append(i)
    return {"edited": edited, "non_surface": non_surface, "core_agreement": core}


def subset(rows: list[dict], idx: list[int]) -> tuple[list[str], list[str], list[str]]:
    src = [rows[i]["source"] for i in idx]
    hyp = [rows[i]["hypothesis"] for i in idx]
    ref = [rows[i]["reference"] for i in idx]
    return src, hyp, ref


# --- sanity check ----------------------------------------------------------

def sanity_check_full_set(test_rows: list[dict]) -> None:
    """Re-score the full set and compare F0.5 to the published eval_test.json.

    A mismatch means the hypotheses rows are misaligned with test.jsonl, which
    would invalidate every subset number below.
    """
    print("=" * 72)
    print("SANITY CHECK: reproduce published full-set F0.5 (word-level scorer)")
    print("=" * 72)
    n_test = len(test_rows)
    for model in ("baseline", "morphaware"):
        for seed in SEEDS:
            run = f"{model}_seed{seed}"
            rows = load_run(run)
            if len(rows) != n_test:
                print(f"  [WARN] {run}: {len(rows)} hyp rows != {n_test} test rows")
            src = [r["source"] for r in rows]
            hyp = [r["hypothesis"] for r in rows]
            ref = [r["reference"] for r in rows]
            m = evaluate_corpus(src, hyp, ref)
            published = json.loads(
                (MULTISEED_RESULTS / run / "eval_test.json").read_text(encoding="utf-8")
            )
            delta = abs(m.f05 - published["f05"])
            flag = "OK" if delta < 5e-3 else "MISMATCH"
            print(f"  {run:22s} recomputed F0.5={m.f05:.4f}  "
                  f"published={published['f05']:.4f}  |d|={delta:.4f}  [{flag}]")
    print()


# --- per-seed subset scoring ----------------------------------------------

def score_subset(name: str, idx: list[int]) -> dict:
    print("=" * 72)
    print(f"SUBSET: {name}  (n={len(idx)} edited pairs)")
    print("=" * 72)
    result: dict = {"name": name, "n": len(idx), "per_seed": {}}
    for model in ("baseline", "morphaware"):
        result["per_seed"][model] = []
        for seed in SEEDS:
            rows = load_run(f"{model}_seed{seed}")
            src, hyp, ref = subset(rows, idx)
            m = evaluate_corpus(src, hyp, ref)
            result["per_seed"][model].append(
                {"seed": seed, "f05": m.f05, "p": m.precision, "r": m.recall,
                 "tp": m.tp, "fp": m.fp, "fn": m.fn}
            )
            print(f"  {model:11s} seed{seed:<4d} F0.5={m.f05:.4f}  "
                  f"P={m.precision:.4f}  R={m.recall:.4f}  "
                  f"(TP={m.tp} FP={m.fp} FN={m.fn})")
    # means
    for model in ("baseline", "morphaware"):
        vals = [d["f05"] for d in result["per_seed"][model]]
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        result["per_seed"][model + "_mean"] = mean
        result["per_seed"][model + "_std"] = var ** 0.5
        print(f"  {model:11s} MEAN      F0.5={mean:.4f} +/- {var ** 0.5:.4f}")
    print()
    return result


def pooled_bootstrap(name: str, idx: list[int]) -> dict:
    """3-seed pooled span-level paired bootstrap, morphaware (A) vs baseline (B).

    A positive delta means morphaware > baseline. Mirrors the chapter's pooled
    n=1941 procedure, restricted to the subset.
    """
    src_all: list[str] = []
    morph_all: list[str] = []
    base_all: list[str] = []
    ref_all: list[str] = []
    for seed in SEEDS:
        b_rows = load_run(f"baseline_seed{seed}")
        m_rows = load_run(f"morphaware_seed{seed}")
        bs, bh, br = subset(b_rows, idx)
        ms, mh, mr = subset(m_rows, idx)
        src_all.extend(bs)
        base_all.extend(bh)
        morph_all.extend(mh)
        ref_all.extend(br)

    res = paired_bootstrap_f05(
        sources=src_all,
        hypotheses_a=morph_all,   # A = morphaware
        hypotheses_b=base_all,    # B = baseline
        references=ref_all,
        n_resamples=2000,
        seed=42,
        scoring="word",
    )
    print("=" * 72)
    print(f"POOLED BOOTSTRAP (morphaware - baseline) on {name}, B=2000, word-level")
    print("=" * 72)
    print(f"  F0.5 morphaware = {res.f05_a:.4f}")
    print(f"  F0.5 baseline   = {res.f05_b:.4f}")
    print(f"  delta           = {res.delta:+.4f}")
    print(f"  95% CI          = [{res.ci_low:+.4f}, {res.ci_high:+.4f}]")
    print(f"  p-value         = {res.p_value:.4f}")
    print(f"  n (pooled)      = {res.n_sentences}")
    print()
    return {
        "name": name,
        "f05_morphaware": res.f05_a,
        "f05_baseline": res.f05_b,
        "delta_morph_minus_base": res.delta,
        "ci_low": res.ci_low,
        "ci_high": res.ci_high,
        "p_value": res.p_value,
        "n_pooled": res.n_sentences,
    }


def main() -> None:
    test_rows = load_jsonl(TEST_JSONL)
    print(f"Loaded {len(test_rows)} test pairs from {TEST_JSONL.name}\n")

    sanity_check_full_set(test_rows)

    idx_sets = build_index_sets(test_rows)
    print(f"Edited pairs (source != target):        {len(idx_sets['edited'])}")
    print(f"Non-surface edited pairs:               {len(idx_sets['non_surface'])}")
    print(f"Core-agreement edited pairs:            {len(idx_sets['core_agreement'])}\n")

    report: dict = {"subsets": {}, "bootstrap": {}}
    for name in ("edited", "non_surface", "core_agreement"):
        idx = idx_sets[name]
        if not idx:
            print(f"[skip] {name}: empty subset\n")
            continue
        report["subsets"][name] = score_subset(name, idx)
        report["bootstrap"][name] = pooled_bootstrap(name, idx)

    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
