"""Validate SoraniDetector on the multilingual gold-labelled set.

Loads `data/language_id_eval/multilingual_gold.jsonl` (200 sentences,
labelled ckb/arb/fas/kmr), runs `SoraniDetector` on each, and reports:
  - Confusion matrix
  - Precision, recall, F1 for the Sorani class at threshold 0.55
  - Per-language false-positive rate
  - Confidence-score distribution by gold label
  - A precision-recall sweep across thresholds (for R18 evidence)

Writes results to `results/sorani_detector_validation.json`.

Run from sorani-gec:
    python scripts/validate_sorani_detector.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.sorani_detector import SoraniDetector

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "data" / "language_id_eval" / "multilingual_gold.jsonl"
OUT = ROOT / "results" / "sorani_detector_validation.json"


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return round(p, 4), round(r, 4), round(f1, 4)


def main() -> None:
    records = [json.loads(l) for l in GOLD.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"Loaded {len(records)} gold-labelled sentences")
    print(f"Per-language gold counts: {dict(Counter(r['language'] for r in records))}")
    print()

    detector = SoraniDetector(threshold=0.55)

    # Evaluate at default threshold 0.55
    confidences: list[float] = []
    preds_per_lang: dict[str, list[bool]] = defaultdict(list)
    conf_per_lang: dict[str, list[float]] = defaultdict(list)
    tp = fp = fn = tn = 0

    for rec in records:
        gold = rec["language"]
        result = detector.detect(rec["text"])
        confidences.append(result.confidence)
        preds_per_lang[gold].append(result.is_sorani)
        conf_per_lang[gold].append(result.confidence)
        is_gold_sorani = gold == "ckb"
        if result.is_sorani and is_gold_sorani:
            tp += 1
        elif result.is_sorani and not is_gold_sorani:
            fp += 1
        elif not result.is_sorani and is_gold_sorani:
            fn += 1
        else:
            tn += 1

    p55, r55, f55 = _prf(tp, fp, fn)
    accuracy = (tp + tn) / len(records)

    print(f"=== Threshold 0.55 (default) ===")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"  Precision={p55:.4f}  Recall={r55:.4f}  F1={f55:.4f}  Accuracy={accuracy:.4f}")
    print()

    # Per-language false-positive rate (for non-Sorani classes)
    per_lang_fp = {}
    for lang, preds in preds_per_lang.items():
        if lang == "ckb":
            per_lang_fp[lang] = {"recall": round(sum(preds) / len(preds), 4),
                                 "mean_confidence": round(sum(conf_per_lang[lang]) / len(preds), 4)}
        else:
            per_lang_fp[lang] = {"fp_rate": round(sum(preds) / len(preds), 4),
                                 "mean_confidence": round(sum(conf_per_lang[lang]) / len(preds), 4)}

    print("=== Per-language behaviour ===")
    for lang, stats in sorted(per_lang_fp.items()):
        print(f"  {lang}: {stats}")
    print()

    # Threshold sweep
    sweep = {}
    for thr in [0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80]:
        d = SoraniDetector(threshold=thr)
        tp2 = fp2 = fn2 = 0
        for rec in records:
            is_gold_sorani = rec["language"] == "ckb"
            pred = d.detect(rec["text"]).is_sorani
            if pred and is_gold_sorani: tp2 += 1
            elif pred and not is_gold_sorani: fp2 += 1
            elif not pred and is_gold_sorani: fn2 += 1
        p, r, f = _prf(tp2, fp2, fn2)
        sweep[f"{thr:.2f}"] = {"precision": p, "recall": r, "f1": f, "tp": tp2, "fp": fp2, "fn": fn2}

    print("=== Threshold sweep ===")
    print(f"  {'thr':>6}  {'P':>6}  {'R':>6}  {'F1':>6}  {'TP':>4}  {'FP':>4}  {'FN':>4}")
    for thr, m in sweep.items():
        print(f"  {thr:>6}  {m['precision']:>6.4f}  {m['recall']:>6.4f}  {m['f1']:>6.4f}  {m['tp']:>4}  {m['fp']:>4}  {m['fn']:>4}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "gold_set": str(GOLD.relative_to(ROOT).as_posix()),
        "n_sentences": len(records),
        "gold_distribution": dict(Counter(r["language"] for r in records)),
        "threshold_0.55": {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": p55, "recall": r55, "f1": f55, "accuracy": round(accuracy, 4),
        },
        "per_language": per_lang_fp,
        "threshold_sweep": sweep,
        "signal_weights": {"script": 0.35, "function_words": 0.40, "morphology": 0.25},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nResults written to {OUT.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
