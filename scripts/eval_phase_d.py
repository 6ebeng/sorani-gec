"""Batch evaluation of all Phase D checkpoints on the test split.

Mirrors the training setup exactly:
  - MorphologicalAnalyzer(use_klpt=False, ahmadi_lexicon=SoraniLexicon())
  - max_length=256
  - batch_size=16, num_beams=4

Writes one JSON results file per run to results/phase_d/<run>/eval_test.json.
Prints a summary table at the end.
"""

import json
import logging
import sys
import os
import math
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
# Show only our own logger at INFO
logging.getLogger("__main__").setLevel(logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger(__name__).setLevel(logging.INFO)

from src.evaluation.f05_scorer import evaluate_corpus, evaluate_corpus_span
from src.model.baseline import BaselineGEC
from src.model.morphology_aware import MorphologyAwareGEC, EDGE_TYPE_ORDER
from src.morphology.analyzer import MorphologicalAnalyzer
from src.morphology.features import FeatureExtractor
from src.morphology.lexicon import SoraniLexicon

DATA_DIR = os.environ.get("PHASE_DATA_DIR", "data/splits_v2")
RESULTS_DIR = os.environ.get("PHASE_RESULTS_DIR", "results/phase_d")
MAX_LENGTH = 256
BATCH_SIZE = 16
NUM_BEAMS = 4
BACKBONE = "google/byt5-small"

RUNS = [
    ("baseline_seed42",   False),
    ("baseline_seed123",  False),
    ("baseline_seed777",  False),
    ("morphaware_seed42",  True),
    ("morphaware_seed123", True),
    ("morphaware_seed777", True),
]


def load_test(data_dir: str):
    src_list, tgt_list = [], []
    with open(f"{data_dir}/test.jsonl", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            src_list.append(rec["source"])
            tgt_list.append(rec["target"])
    return src_list, tgt_list


def build_morphaware(lexicon, feature_vocab):
    return MorphologyAwareGEC(
        model_name=BACKBONE,
        feature_vocab_size=max(len(feature_vocab), 1),
        num_agreement_types=len(EDGE_TYPE_ORDER) + 1,
        max_length=MAX_LENGTH,
    )


def build_baseline():
    return BaselineGEC(model_name=BACKBONE, max_length=MAX_LENGTH)


def run_eval(run_name: str, is_morphaware: bool,
             srcs: list, tgts: list,
             device, lexicon, analyzer, feature_extractor, feature_vocab):
    ckpt_path = f"{RESULTS_DIR}/{run_name}/best_model.pt"
    out_path = f"{RESULTS_DIR}/{run_name}/eval_test.json"

    print(f"\n{'='*60}")
    print(f"  {run_name}")
    print(f"{'='*60}")

    if is_morphaware:
        model = build_morphaware(lexicon, feature_vocab)
    else:
        model = build_baseline()

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

    metrics = evaluate_corpus(srcs, hypotheses, tgts)
    span_metrics, _span_by_type = evaluate_corpus_span(srcs, hypotheses, tgts)
    result = {
        "run": run_name,
        # Headline is the span-aware (position-aware) scorer; the word-level
        # numbers are kept under word_* for backward comparison only.
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
    print(f"  span F0.5={span_metrics.f05:.4f}  P={span_metrics.precision:.4f}  R={span_metrics.recall:.4f}  (word F0.5={metrics.f05:.4f})")
    print(f"  Saved -> {out_path}")

    # Free VRAM
    del model
    torch.cuda.empty_cache()
    return result


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    srcs, tgts = load_test(DATA_DIR)
    print(f"Test set: {len(srcs)} sentences")

    lexicon = SoraniLexicon()
    analyzer = MorphologicalAnalyzer(use_klpt=False, ahmadi_lexicon=lexicon)
    feature_vocab = analyzer.build_feature_vocabulary()
    feature_extractor = FeatureExtractor(analyzer)
    print(f"Feature vocab size: {len(feature_vocab)}")

    all_results = []
    for run_name, is_morphaware in RUNS:
        res = run_eval(run_name, is_morphaware, srcs, tgts, device,
                       lexicon, analyzer, feature_extractor, feature_vocab)
        all_results.append(res)

    # Summary table
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Run':<25} {'F0.5':>8} {'P':>8} {'R':>8} {'TP':>6} {'FP':>7} {'FN':>6}")
    print("-" * 70)
    for r in all_results:
        print(f"{r['run']:<25} {r['f05']:>8.4f} {r['precision']:>8.4f} {r['recall']:>8.4f} {r['tp']:>6} {r['fp']:>7} {r['fn']:>6}")

    # Mean ± std by model type
    for label, keys in [("Baseline", ["baseline_seed42","baseline_seed123","baseline_seed777"]),
                         ("Morphaware", ["morphaware_seed42","morphaware_seed123","morphaware_seed777"])]:
        vals = [r["f05"] for r in all_results if r["run"] in keys]
        mean = sum(vals) / len(vals)
        std = math.sqrt(sum((v - mean)**2 for v in vals) / len(vals))
        print(f"\n{label}: mean F0.5 = {mean:.4f} ± {std:.4f}  (seeds: {', '.join(f'{v:.4f}' for v in vals)})")

    # Save combined
    with open(f"{RESULTS_DIR}/eval_summary.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nFull summary -> {RESULTS_DIR}/eval_summary.json")


if __name__ == "__main__":
    main()
