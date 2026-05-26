"""Automated proxy evaluation in lieu of human raters (R8).

The reviewer report (R8) asks for a 60-pair human grammaticality study with
two native Sorani speakers and Cohen's kappa. Native-speaker access is
unavailable within the thesis timeline, so this script produces an automated
stand-in. The thesis discusses the substitution explicitly in Ch.7 and lists
the full human study under Ch.9 future work.

What this script does:

1. Samples 60 (source, corrected) pairs from the real baseline test-time
   predictions at ``results/metrics_remote/baseline/evaluation_pairs.jsonl``,
   stratified by character-level edit ratio so that trivial, mild, and heavy
   edits are all represented.
2. Applies two automated raters that target genuinely independent signals:
   - **AUTO-A — rule-based grammaticality**: runs ``AgreementChecker`` on the
     corrected sentence; the fraction of passing checks is binned into a
     1-5 ordinal scale.
   - **AUTO-B — surface fluency**: scores the corrected output on character
     repetition, length stability, and non-Kurdish character ratio; bins to
     a 1-5 ordinal scale.
3. Writes ``ratings_<id>.jsonl`` for each rater and the 60-pair selection to
   ``evaluation_pairs.jsonl``, all under ``results/human_eval/``.
4. Computes Cohen's kappa via ``src.evaluation.inter_rater``.

The kappa here measures how consistent two automated quality signals are on
real model outputs — it is *not* a substitute for native-speaker consensus.

Run from ``Implementation/sorani-gec``::

    python scripts/auto_rate_eval_pairs.py
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.agreement_accuracy import AgreementChecker
from src.evaluation.inter_rater import compute_inter_rater_agreement

PREDS = ROOT / "results" / "metrics_remote" / "baseline" / "evaluation_pairs.jsonl"
EVAL_DIR = ROOT / "results" / "human_eval"
PAIRS_OUT = EVAL_DIR / "evaluation_pairs.jsonl"

RATER_A_ID = "auto_agreement_rules"
RATER_B_ID = "auto_surface_fluency"

N_PAIRS = 60
SEED = 42
SORANI_RANGE = (0x0600, 0x06FF)  # Arabic block — Sorani uses this range


def _cer(a: str, b: str) -> float:
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


def _stratified_sample(pairs: list[dict], n: int, rng: random.Random) -> list[dict]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for p in pairs:
        cer = _cer(p["source"].strip(), p["corrected"].strip())
        if cer == 0.0:
            b = "no_edit"
        elif cer <= 0.05:
            b = "trivial"
        elif cer <= 0.20:
            b = "mild"
        elif cer <= 0.50:
            b = "heavy"
        else:
            b = "rewrite"
        buckets[b].append(p)
    total = sum(len(v) for v in buckets.values())
    selected, leftover = [], []
    for bname, items in buckets.items():
        rng.shuffle(items)
        quota = max(1, round(len(items) / total * n))
        selected.extend(items[:quota])
        leftover.extend(items[quota:])
    if len(selected) < n:
        rng.shuffle(leftover)
        selected.extend(leftover[: n - len(selected)])
    rng.shuffle(selected)
    return selected[:n]


# ── Rater A: rule-based grammaticality ──────────────────────────────────────

def _build_rater_a():
    checker = AgreementChecker()

    def _rate(_source: str, corrected: str) -> int:
        text = corrected.strip()
        if not text:
            return 1
        result = checker.check_sentence(text)
        acc = result.accuracy
        if acc >= 0.99:
            return 5
        if acc >= 0.85:
            return 4
        if acc >= 0.70:
            return 3
        if acc >= 0.50:
            return 2
        return 1

    return _rate


# ── Rater B: surface fluency ────────────────────────────────────────────────

def _max_repeat_run(text: str) -> int:
    best, cur, prev = 1, 1, None
    for ch in text:
        if ch == prev:
            cur += 1
            best = max(best, cur)
        else:
            cur = 1
        prev = ch
    return best if text else 0


def _non_kurdish_ratio(text: str) -> float:
    """Fraction of non-whitespace chars outside the Arabic Unicode block."""
    chars = [c for c in text if not c.isspace() and not c.isdigit() and not c in ".,؛؟،!?:"]
    if not chars:
        return 1.0
    ok = sum(1 for c in chars if SORANI_RANGE[0] <= ord(c) <= SORANI_RANGE[1])
    return 1.0 - ok / len(chars)


def _rate_b_fluency(source: str, corrected: str) -> int:
    text = corrected.strip()
    src = source.strip()
    if not text:
        return 1
    # Length stability vs source
    len_dev = abs(len(text) - len(src)) / max(len(text), len(src), 1)
    repeat = _max_repeat_run(text)
    foreign = _non_kurdish_ratio(text)

    # Penalty score: 0 = clean, higher = worse
    pen = 0
    if foreign > 0.20:
        pen += 3
    elif foreign > 0.10:
        pen += 2
    elif foreign > 0.05:
        pen += 1
    if repeat >= 6:
        pen += 3
    elif repeat >= 4:
        pen += 2
    elif repeat >= 3:
        pen += 1
    if len_dev > 0.50:
        pen += 2
    elif len_dev > 0.20:
        pen += 1

    if pen == 0:
        return 5
    if pen == 1:
        return 4
    if pen == 2:
        return 3
    if pen <= 4:
        return 2
    return 1


# ── orchestration ───────────────────────────────────────────────────────────

def _write_ratings(rater_id: str, rate_fn, pairs: list[dict]) -> Path:
    out = EVAL_DIR / f"ratings_{rater_id}.jsonl"
    with out.open("w", encoding="utf-8", newline="\n") as fh:
        for p in pairs:
            rating = rate_fn(p["source"], p["corrected"])
            rec = {
                "rater_id": rater_id,
                "source": p["source"],
                "corrected": p["corrected"],
                "rating": str(rating),
                "annotator_type": "automated_proxy",
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return out


def main() -> None:
    if not PREDS.exists():
        raise SystemExit(f"Baseline predictions not found: {PREDS}")
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    # Clean previously generated rating files so only the new ones contribute
    for old in EVAL_DIR.glob("ratings_*.jsonl"):
        old.unlink()

    preds = [json.loads(l) for l in PREDS.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"Loaded {len(preds)} baseline predictions")

    rng = random.Random(SEED)
    sample = _stratified_sample(preds, N_PAIRS, rng)
    PAIRS_OUT.write_text(
        "\n".join(json.dumps({"source": p["source"], "corrected": p["corrected"]}, ensure_ascii=False) for p in sample) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(sample)} pairs to {PAIRS_OUT.relative_to(ROOT)}")

    rater_a = _build_rater_a()
    a_path = _write_ratings(RATER_A_ID, rater_a, sample)
    b_path = _write_ratings(RATER_B_ID, _rate_b_fluency, sample)
    print(f"Wrote ratings: {a_path.name}, {b_path.name}")

    a_dist = Counter(str(rater_a(p["source"], p["corrected"])) for p in sample)
    b_dist = Counter(str(_rate_b_fluency(p["source"], p["corrected"])) for p in sample)
    print(f"\nRater A ({RATER_A_ID}) distribution:")
    for k in sorted(a_dist):
        print(f"  rating={k}: {a_dist[k]}")
    print(f"Rater B ({RATER_B_ID}) distribution:")
    for k in sorted(b_dist):
        print(f"  rating={k}: {b_dist[k]}")

    results = compute_inter_rater_agreement(EVAL_DIR)
    print("\n=== Inter-Rater Agreement (automated proxy) ===")
    print(json.dumps(results, ensure_ascii=False, indent=2))

    summary = {
        "evaluation_type": "automated_proxy_in_lieu_of_humans",
        "source_predictions": str(PREDS.relative_to(ROOT)),
        "n_pairs": len(sample),
        "raters": {
            RATER_A_ID: "AgreementChecker on corrected; passing-checks fraction binned to 1-5",
            RATER_B_ID: "Surface fluency: char-repeat + length stability + non-Kurdish char ratio; penalty binned to 1-5",
        },
        "rater_A_distribution": dict(a_dist),
        "rater_B_distribution": dict(b_dist),
        "agreement": results,
        "caveat": (
            "Both raters are automated and target different signals (rule-based "
            "grammaticality vs surface fluency). The kappa value reflects how "
            "consistent two independent automated quality signals are on real "
            "model output — it does not substitute for native-speaker consensus. "
            "Full human evaluation with two Sorani native speakers is documented "
            "as future work in thesis Ch.9."
        ),
    }
    (EVAL_DIR / "automated_proxy_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nSummary written to results/human_eval/automated_proxy_summary.json")


if __name__ == "__main__":
    main()
