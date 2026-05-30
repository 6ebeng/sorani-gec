"""Dump per-sentence hypotheses for every Phase D checkpoint.

Reviewer item R9: the sentence-level paired bootstrap needs each model's
output on every test sentence, not just the aggregate F0.5. This script
mirrors eval_phase_d.py exactly (same analyzer, max_length, beams) but writes
results/phase_d/<run>/hypotheses.jsonl with {source, hypothesis, reference}
per line. Those files are small and can be downloaded for local significance
testing.
"""

import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

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
    srcs, tgts = [], []
    with open(f"{data_dir}/test.jsonl", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            srcs.append(rec["source"])
            tgts.append(rec["target"])
    return srcs, tgts


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    srcs, tgts = load_test(DATA_DIR)
    print(f"Test set: {len(srcs)} sentences")

    lexicon = SoraniLexicon()
    analyzer = MorphologicalAnalyzer(use_klpt=False, ahmadi_lexicon=lexicon)
    feature_vocab = analyzer.build_feature_vocabulary()
    feature_extractor = FeatureExtractor(analyzer)

    for run_name, is_morphaware in RUNS:
        ckpt_path = f"{RESULTS_DIR}/{run_name}/best_model.pt"
        out_path = f"{RESULTS_DIR}/{run_name}/hypotheses.jsonl"
        print(f"\n=== {run_name} ===")

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

        hyps = []
        t0 = time.time()
        with torch.no_grad():
            for i in range(0, len(srcs), BATCH_SIZE):
                batch = srcs[i:i + BATCH_SIZE]
                if is_morphaware:
                    out = model.correct_batch(batch, analyzer, feature_extractor, num_beams=NUM_BEAMS)
                else:
                    out = model.correct_batch(batch, num_beams=NUM_BEAMS)
                hyps.extend(out)
        print(f"  {len(hyps)} hyps in {time.time()-t0:.1f}s")

        with open(out_path, "w", encoding="utf-8") as f:
            for s, h, t in zip(srcs, hyps, tgts):
                f.write(json.dumps({"source": s, "hypothesis": h, "reference": t}, ensure_ascii=False) + "\n")
        print(f"  Saved -> {out_path}")

        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
