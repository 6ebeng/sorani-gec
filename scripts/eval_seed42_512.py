"""Evaluate baseline_seed42 and morphaware_seed42 at max_length=512.

Used for the clean-campaign results (results/campaign_3_clean_final, formerly
results/phase2_clean) where models were trained at --max-length 512.
Run from the repo root:
  CAMPAIGN_RESULTS_DIR=results/campaign_3_clean_final CAMPAIGN_DATA_DIR=data/splits_scaled \
      python3 scripts/eval_seed42_512.py
"""
import json
import logging
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logging.getLogger("__main__").setLevel(logging.INFO)

from src.evaluation.f05_scorer import evaluate_corpus, evaluate_corpus_span
from src.model.baseline import BaselineGEC
from src.model.morphology_aware import MorphologyAwareGEC, EDGE_TYPE_ORDER
from src.morphology.analyzer import MorphologicalAnalyzer
from src.morphology.features import FeatureExtractor
from src.morphology.lexicon import SoraniLexicon

DATA_DIR     = os.environ.get("CAMPAIGN_DATA_DIR", "data/splits_scaled")
RESULTS_DIR  = os.environ.get("CAMPAIGN_RESULTS_DIR", "results/campaign_3_clean_final")
MAX_LENGTH   = 512
BATCH_SIZE   = 8
NUM_BEAMS    = 4
BACKBONE     = "google/byt5-small"

RUNS = [
    ("baseline_seed42",   False),
    ("morphaware_seed42", True),
]


def load_test(data_dir):
    srcs, tgts = [], []
    with open(f"{data_dir}/test.jsonl", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            srcs.append(rec["source"])
            tgts.append(rec["target"])
    return srcs, tgts


def run_eval(run_name, is_morphaware, srcs, tgts, device,
             lexicon, analyzer, feature_extractor, feature_vocab):
    ckpt_path = f"{RESULTS_DIR}/{run_name}/best_model.pt"
    out_path  = f"{RESULTS_DIR}/{run_name}/eval_test_512.json"

    print(f"\n{'='*60}\n  {run_name}\n{'='*60}")

    if is_morphaware:
        model = MorphologyAwareGEC(
            model_name=BACKBONE,
            feature_vocab_size=max(len(feature_vocab), 1),
            num_agreement_types=len(EDGE_TYPE_ORDER) + 1,
            max_length=MAX_LENGTH,
        )
    else:
        model = BaselineGEC(model_name=BACKBONE, max_length=MAX_LENGTH)

    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    print(f"  Loaded {ckpt_path}")

    hypotheses = []
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(srcs), BATCH_SIZE):
            batch = srcs[i:i + BATCH_SIZE]
            if is_morphaware:
                hyps = model.correct_batch(batch, analyzer, feature_extractor, num_beams=NUM_BEAMS)
            else:
                hyps = model.correct_batch(batch, num_beams=NUM_BEAMS)
            hypotheses.extend(hyps)
            if (i // BATCH_SIZE) % 5 == 0:
                print(f"  {min(i+BATCH_SIZE, len(srcs))}/{len(srcs)} sentences ...", flush=True)
    elapsed = time.time() - t0

    metrics      = evaluate_corpus(srcs, hypotheses, tgts)
    span_metrics, _ = evaluate_corpus_span(srcs, hypotheses, tgts)

    result = {
        "run": run_name,
        "max_length": MAX_LENGTH,
        "f05": span_metrics.f05,
        "precision": span_metrics.precision,
        "recall": span_metrics.recall,
        "tp": span_metrics.tp,
        "fp": span_metrics.fp,
        "fn": span_metrics.fn,
        "word_f05": metrics.f05,
        "word_precision": metrics.precision,
        "word_recall": metrics.recall,
        "word_tp": metrics.tp,
        "word_fp": metrics.fp,
        "word_fn": metrics.fn,
        "elapsed_sec": round(elapsed, 1),
        "n_sentences": len(srcs),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"  span F0.5={span_metrics.f05:.4f}  P={span_metrics.precision:.4f}  "
          f"R={span_metrics.recall:.4f}  (word F0.5={metrics.f05:.4f})")
    print(f"  TP={span_metrics.tp}  FP={span_metrics.fp}  FN={span_metrics.fn}")
    print(f"  Saved -> {out_path}")

    del model
    torch.cuda.empty_cache()
    return result, hypotheses


def run_bootstrap(base_hyps, morph_hyps, srcs, tgts, n_bootstrap=10000, seed=0):
    """Paired bootstrap: is baseline better than morphaware (or vice versa)?"""
    import random
    rng = random.Random(seed)
    n = len(srcs)

    def f05(tp, fp, fn):
        denom = (1.25 * tp) + fp + (0.25 * fn)
        return (1.25 * tp) / denom if denom > 0 else 0.0

    # Per-sentence TP/FP/FN for each model
    from src.evaluation.f05_scorer import evaluate_corpus_span
    def per_sent_scores(hyps):
        scores = []
        for s, h, t in zip(srcs, hyps, tgts):
            m, _ = evaluate_corpus_span([s], [h], [t])
            scores.append((m.tp, m.fp, m.fn))
        return scores

    print("\nComputing per-sentence scores for bootstrap (slow ~2-3 min)...")
    base_scores  = per_sent_scores(base_hyps)
    morph_scores = per_sent_scores(morph_hyps)

    base_tp  = sum(s[0] for s in base_scores)
    base_fp  = sum(s[1] for s in base_scores)
    base_fn  = sum(s[2] for s in base_scores)
    morph_tp = sum(s[0] for s in morph_scores)
    morph_fp = sum(s[1] for s in morph_scores)
    morph_fn = sum(s[2] for s in morph_scores)
    base_f05  = f05(base_tp,  base_fp,  base_fn)
    morph_f05 = f05(morph_tp, morph_fp, morph_fn)
    delta_obs = morph_f05 - base_f05

    count = 0
    for _ in range(n_bootstrap):
        idxs = [rng.randrange(n) for _ in range(n)]
        b_tp = sum(base_scores[i][0]  for i in idxs)
        b_fp = sum(base_scores[i][1]  for i in idxs)
        b_fn = sum(base_scores[i][2]  for i in idxs)
        m_tp = sum(morph_scores[i][0] for i in idxs)
        m_fp = sum(morph_scores[i][1] for i in idxs)
        m_fn = sum(morph_scores[i][2] for i in idxs)
        delta_b = f05(m_tp, m_fp, m_fn) - f05(b_tp, b_fp, b_fn)
        if delta_obs >= 0 and delta_b >= delta_obs * 2:
            count += 1
        elif delta_obs < 0 and delta_b <= delta_obs * 2:
            count += 1

    p_val = count / n_bootstrap
    return {
        "baseline_f05":  base_f05,
        "morphaware_f05": morph_f05,
        "delta": delta_obs,
        "p_value": p_val,
        "n_bootstrap": n_bootstrap,
        "significant_p05": p_val < 0.05,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"MAX_LENGTH: {MAX_LENGTH}")

    srcs, tgts = load_test(DATA_DIR)
    print(f"Test set: {len(srcs)} sentences")

    lexicon           = SoraniLexicon()
    analyzer          = MorphologicalAnalyzer(use_klpt=False, ahmadi_lexicon=lexicon)
    feature_vocab     = analyzer.build_feature_vocabulary()
    feature_extractor = FeatureExtractor(analyzer)
    print(f"Feature vocab size: {len(feature_vocab)}")

    all_results = []
    all_hyps    = {}
    for run_name, is_morph in RUNS:
        res, hyps = run_eval(run_name, is_morph, srcs, tgts, device,
                             lexicon, analyzer, feature_extractor, feature_vocab)
        all_results.append(res)
        all_hyps[run_name] = hyps

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY  (span F0.5, max_length=512)")
    print(f"{'='*60}")
    print(f"{'Run':<25} {'F0.5':>8} {'P':>8} {'R':>8} {'TP':>6} {'FP':>7} {'FN':>6}")
    print("-" * 70)
    for r in all_results:
        print(f"{r['run']:<25} {r['f05']:>8.4f} {r['precision']:>8.4f} "
              f"{r['recall']:>8.4f} {r['tp']:>6} {r['fp']:>7} {r['fn']:>6}")

    # Bootstrap
    boot = run_bootstrap(
        all_hyps["baseline_seed42"],
        all_hyps["morphaware_seed42"],
        srcs, tgts,
    )
    print(f"\nBootstrap ({boot['n_bootstrap']:,} resamples):")
    print(f"  Baseline   F0.5 = {boot['baseline_f05']:.4f}")
    print(f"  Morphaware F0.5 = {boot['morphaware_f05']:.4f}")
    print(f"  Delta            = {boot['delta']:+.4f}")
    print(f"  p-value          = {boot['p_value']:.4f}  "
          f"({'significant' if boot['significant_p05'] else 'NOT significant'} at p<0.05)")

    # Save combined
    summary = {"runs": all_results, "bootstrap": boot}
    out = f"{RESULTS_DIR}/eval_summary_512.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nFull results -> {out}")


if __name__ == "__main__":
    main()
