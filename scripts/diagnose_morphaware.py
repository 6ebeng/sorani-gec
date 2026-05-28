"""Diagnose morphaware frozen-output bug: inspect actual generations."""
import sys
import torch
sys.path.insert(0, "src")
from src.model.morphology_aware import MorphologyAwareGEC, EDGE_TYPE_ORDER
from src.model.baseline import BaselineGEC as BaselineByT5GEC
from src.morphology.analyzer import MorphologicalAnalyzer
from src.morphology.features import FeatureExtractor
from src.morphology.lexicon import SoraniLexicon

device = torch.device("cuda")

lexicon = SoraniLexicon()
analyzer = MorphologicalAnalyzer(use_klpt=False, ahmadi_lexicon=lexicon)
feature_vocab = analyzer.build_feature_vocabulary()
fe = FeatureExtractor(analyzer)
num_agr_types = len(EDGE_TYPE_ORDER) + 1


def make_morph():
    return MorphologyAwareGEC(
        model_name="google/byt5-small",
        feature_vocab_size=max(len(feature_vocab), 1),
        num_agreement_types=num_agr_types,
    ).to(device)

import json
srcs, tgts = [], []
with open("data/splits_v2/dev.jsonl", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 5:
            break
        obj = json.loads(line)
        srcs.append(obj.get("source") or obj.get("src") or obj.get("source_text"))
        tgts.append(obj.get("target") or obj.get("tgt") or obj.get("target_text"))

print("=" * 60)
print("MORPHAWARE seed=42")
print("=" * 60)
m = make_morph()
ckpt = torch.load("results/phase_d/morphaware_seed42/best_model.pt", map_location=device, weights_only=True)
m.load_state_dict(ckpt["model_state_dict"])
m.eval()
hyps = m.correct_batch(srcs, analyzer, fe, num_beams=4)
for s, h, t in zip(srcs, hyps, tgts):
    print("SRC:", repr(s[:100]))
    print("HYP:", repr(h[:100]))
    print("TGT:", repr(t[:100]))
    print("---")

print()
print("=" * 60)
print("MORPHAWARE seed=777 (different weights)")
print("=" * 60)
m2 = make_morph()
ckpt2 = torch.load("results/phase_d/morphaware_seed777/best_model.pt", map_location=device, weights_only=True)
m2.load_state_dict(ckpt2["model_state_dict"])
m2.eval()
hyps2 = m2.correct_batch(srcs, analyzer, fe, num_beams=4)
for s, h in zip(srcs, hyps2):
    print("SRC:", repr(s[:100]))
    print("HYP:", repr(h[:100]))
    print("---")

print()
print("Identical between seeds?", hyps == hyps2)

print()
print("=" * 60)
print("BASELINE seed=42 (sanity check)")
print("=" * 60)
b = BaselineByT5GEC("google/byt5-small").to(device)
ckpt_b = torch.load("results/phase_d/baseline_seed42/best_model.pt", map_location=device, weights_only=True)
b.load_state_dict(ckpt_b["model_state_dict"])
b.eval()
hyps_b = b.correct_batch(srcs, num_beams=4)
for s, h, t in zip(srcs, hyps_b, tgts):
    print("SRC:", repr(s[:100]))
    print("HYP:", repr(h[:100]))
    print("TGT:", repr(t[:100]))
    print("---")
