"""Analyse the Phase F human-evaluation ratings (R6 + R15).

Reads the per-annotator ``ratings_<id>.jsonl`` files written by the Gradio app
and the hidden ``evaluation_pairs_manifest.jsonl`` produced by
``build_eval_pairs.py``. Produces two things the reviewer asked for:

  R6  — inter-annotator agreement: Cohen's kappa and percentage agreement for
        every annotator pair.
  R15 — validation of the 14-check agreement metric against human grammaticality
        judgement: Kendall's tau-b (human ordinal vs metric accuracy) and Cohen's
        kappa (human binary grammaticality vs metric pass/fail).

It also reports per-system mean human grammaticality (baseline vs morphology-
aware), so the human study can be read against the F0.5 ordering.

Automated-proxy raters (ids starting ``auto_``) are excluded by default so the
study reflects real annotators; pass ``--include-proxy`` to include them.

Writes ``results/human_eval/metric_validation.json``.

Run from ``Implementation/sorani-gec``::

    python scripts/analyze_human_eval.py
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.inter_rater import cohens_kappa, percentage_agreement

EVAL_DIR = ROOT / "results" / "human_eval"
MANIFEST = EVAL_DIR / "evaluation_pairs_manifest.jsonl"
OUT = EVAL_DIR / "metric_validation.json"

# Sorani rating -> ordinal grammaticality (higher = more grammatical).
# نادیار (unsure) / بازدان (skip) carry no judgement.
_ORDINAL = {
    "دروست": 3,
    "بەشێکی دروست": 2,
    "هەڵە": 1,
}
_SKIP = {"نادیار", "بازدان"}
# A human calls an output grammatical only when it is fully correct.
_BINARY_GRAMMATICAL = {"دروست"}


def _to_ordinal(rating: str) -> int | None:
    """Map a stored rating to an ordinal score, or None if it carries no signal."""
    rating = str(rating).strip()
    if rating in _ORDINAL:
        return _ORDINAL[rating]
    if rating in _SKIP:
        return None
    # Numeric 1-5 scale (automated proxy raters).
    if rating.isdigit():
        return int(rating)
    return None


def _to_binary(rating: str) -> int | None:
    """1 = human judges fully grammatical, 0 = not, None = no signal."""
    rating = str(rating).strip()
    if rating in _SKIP:
        return None
    if rating in _ORDINAL:
        return 1 if rating in _BINARY_GRAMMATICAL else 0
    if rating.isdigit():
        return 1 if int(rating) >= 4 else 0
    return None


def _kendall_tau_b(x: list[float], y: list[float]) -> tuple[float, int]:
    """Kendall's tau-b with tie correction. Returns (tau, n_pairs)."""
    n = len(x)
    if n < 2:
        return 0.0, n
    concordant = discordant = 0
    tx = ty = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            s = (dx > 0) - (dx < 0)
            t = (dy > 0) - (dy < 0)
            prod = s * t
            if prod > 0:
                concordant += 1
            elif prod < 0:
                discordant += 1
            else:
                if dx == 0:
                    tx += 1
                if dy == 0:
                    ty += 1
    n0 = n * (n - 1) / 2
    denom = ((n0 - tx) * (n0 - ty)) ** 0.5
    if denom == 0:
        return 0.0, n
    return (concordant - discordant) / denom, n


def _load_ratings(include_proxy: bool) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for f in sorted(EVAL_DIR.glob("ratings_*.jsonl")):
        rater = f.stem.replace("ratings_", "")
        if not include_proxy and rater.startswith("auto_"):
            continue
        rows = [json.loads(ln) for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if rows:
            out[rater] = rows
    return out


def _load_manifest() -> dict[tuple[str, str], dict]:
    if not MANIFEST.exists():
        raise SystemExit(f"Manifest not found: {MANIFEST}. Run build_eval_pairs.py first.")
    index = {}
    for ln in MANIFEST.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        rec = json.loads(ln)
        index[(rec["source"], rec["corrected"])] = rec
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse Phase F human evaluation")
    parser.add_argument("--include-proxy", action="store_true",
                        help="Include automated-proxy raters (ids starting 'auto_')")
    args = parser.parse_args()

    ratings = _load_ratings(args.include_proxy)
    if not ratings:
        raise SystemExit(
            "No annotator ratings found. Annotators rate in the web app first; "
            "ratings land in results/human_eval/ratings_<id>.jsonl."
        )
    manifest = _load_manifest()

    # ── R6: inter-annotator agreement (raw 5-category rating) ───────────────
    rater_ids = sorted(ratings)
    iaa = {}
    for i in range(len(rater_ids)):
        for j in range(i + 1, len(rater_ids)):
            ra, rb = rater_ids[i], rater_ids[j]
            key = lambda r: (r.get("source", ""), r.get("corrected", ""))
            map_b = {key(r): r["rating"] for r in ratings[rb]}
            la, lb = [], []
            for r in ratings[ra]:
                k = key(r)
                if k in map_b:
                    la.append(str(r["rating"]))
                    lb.append(str(map_b[k]))
            if la:
                iaa[f"{ra} vs {rb}"] = {
                    "cohens_kappa": round(cohens_kappa(la, lb), 4),
                    "percent_agreement": round(percentage_agreement(la, lb), 4),
                    "n_overlap": len(la),
                }

    # ── Aggregate per-pair human judgement across annotators ────────────────
    pair_ordinals: dict[tuple[str, str], list[int]] = defaultdict(list)
    pair_binaries: dict[tuple[str, str], list[int]] = defaultdict(list)
    for rows in ratings.values():
        for r in rows:
            k = (r.get("source", ""), r.get("corrected", ""))
            o = _to_ordinal(r.get("rating", ""))
            b = _to_binary(r.get("rating", ""))
            if o is not None:
                pair_ordinals[k].append(o)
            if b is not None:
                pair_binaries[k].append(b)

    # ── R15: human grammaticality vs the 14-check metric ────────────────────
    human_ord, metric_acc = [], []
    human_bin, metric_pass = [], []
    per_system_ord: dict[str, list[float]] = defaultdict(list)
    matched = 0
    for k, scores in pair_ordinals.items():
        rec = manifest.get(k)
        if rec is None:
            continue
        matched += 1
        mean_ord = sum(scores) / len(scores)
        human_ord.append(mean_ord)
        metric_acc.append(float(rec["agreement_accuracy"]))
        per_system_ord[rec["system"]].append(mean_ord)
        bins = pair_binaries.get(k, [])
        if bins:
            maj = 1 if sum(bins) >= len(bins) / 2 else 0
            human_bin.append(maj)
            metric_pass.append(1 if rec["agreement_pass"] else 0)

    tau, tau_n = _kendall_tau_b(human_ord, metric_acc)
    kappa_metric = cohens_kappa([str(x) for x in human_bin], [str(x) for x in metric_pass]) if human_bin else 0.0
    pct_metric = percentage_agreement([str(x) for x in human_bin], [str(x) for x in metric_pass]) if human_bin else 0.0

    summary = {
        "n_annotators": len(rater_ids),
        "annotators": rater_ids,
        "n_pairs_in_manifest": len(manifest),
        "n_pairs_with_human_signal": matched,
        "inter_annotator_agreement_R6": iaa,
        "metric_validation_R15": {
            "kendall_tau_b_humanordinal_vs_metricaccuracy": round(tau, 4),
            "tau_n_pairs": tau_n,
            "cohens_kappa_humanbinary_vs_metricpass": round(kappa_metric, 4),
            "percent_agreement_humanbinary_vs_metricpass": round(pct_metric, 4),
            "n_binary_pairs": len(human_bin),
        },
        "human_grammaticality_by_system": {
            sys_: round(sum(v) / len(v), 4) for sys_, v in sorted(per_system_ord.items()) if v
        },
        "includes_automated_proxy": args.include_proxy,
        "ordinal_scale": "1=هەڵە (wrong), 2=بەشێکی دروست (partial), 3=دروست (correct)",
    }

    OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nWrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
