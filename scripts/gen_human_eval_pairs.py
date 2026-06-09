"""Generate evaluation_pairs.jsonl for human evaluation from the clean models.

Picks 60 sentences from the test set: 20 where baseline changes something,
20 where morphaware changes something, 20 where both agree on a change.
Each pair: source + corrected (from the model). Annotators rate the correction.

Run on the instance:
  PHASE_RESULTS_DIR=results/phase2_clean PHASE_DATA_DIR=data/splits_scaled \
      python3 scripts/gen_human_eval_pairs.py
"""
import hashlib, json, os, random, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from src.model.baseline import BaselineGEC
from src.model.morphology_aware import MorphologyAwareGEC, EDGE_TYPE_ORDER
from src.morphology.analyzer import MorphologicalAnalyzer
from src.morphology.features import FeatureExtractor
from src.morphology.lexicon import SoraniLexicon

DATA_DIR    = os.environ.get("PHASE_DATA_DIR",    "data/splits_scaled")
RESULTS_DIR = os.environ.get("PHASE_RESULTS_DIR", "results/phase2_clean")
OUT_DIR     = "results/human_eval"
MAX_LENGTH  = 512
BATCH_SIZE  = 8
NUM_BEAMS   = 4
BACKBONE    = "google/byt5-small"
SEED        = 42
N_EACH      = 20   # 20 baseline-only, 20 morph-only, 20 both-agree = 60 total

random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load test set
srcs, tgts = [], []
with open(f"{DATA_DIR}/test.jsonl", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        srcs.append(r["source"])
        tgts.append(r["target"])
print(f"Test set: {len(srcs)} sentences")

# Load morphology
lexicon           = SoraniLexicon()
analyzer          = MorphologicalAnalyzer(use_klpt=False, ahmadi_lexicon=lexicon)
feature_vocab     = analyzer.build_feature_vocabulary()
feature_extractor = FeatureExtractor(analyzer)

# Load baseline
baseline = BaselineGEC(model_name=BACKBONE, max_length=MAX_LENGTH)
state = torch.load(f"{RESULTS_DIR}/baseline_seed42/best_model.pt", map_location=device, weights_only=True)
if "model_state_dict" in state: state = state["model_state_dict"]
baseline.load_state_dict(state)
baseline = baseline.to(device)
baseline.eval()
print("Baseline loaded.")

# Load morphaware
morph = MorphologyAwareGEC(
    model_name=BACKBONE,
    feature_vocab_size=max(len(feature_vocab), 1),
    num_agreement_types=len(EDGE_TYPE_ORDER) + 1,
    max_length=MAX_LENGTH,
)
state = torch.load(f"{RESULTS_DIR}/morphaware_seed42/best_model.pt", map_location=device, weights_only=True)
if "model_state_dict" in state: state = state["model_state_dict"]
morph.load_state_dict(state)
morph = morph.to(device)
morph.eval()
print("Morphaware loaded.")

# Run inference
base_hyps, morph_hyps = [], []
with torch.no_grad():
    for i in range(0, len(srcs), BATCH_SIZE):
        batch = srcs[i:i+BATCH_SIZE]
        base_hyps.extend(baseline.correct_batch(batch, num_beams=NUM_BEAMS))
        morph_hyps.extend(morph.correct_batch(batch, analyzer, feature_extractor, num_beams=NUM_BEAMS))
        if i % 80 == 0:
            print(f"  {min(i+BATCH_SIZE, len(srcs))}/{len(srcs)} ...", flush=True)

print("Inference done.")

# Categorise
base_changed  = [i for i in range(len(srcs)) if base_hyps[i] != srcs[i]]
morph_changed = [i for i in range(len(srcs)) if morph_hyps[i] != srcs[i]]
base_set  = set(base_changed)
morph_set = set(morph_changed)
only_base  = [i for i in base_changed  if i not in morph_set]
only_morph = [i for i in morph_changed if i not in base_set]
both       = [i for i in base_changed  if i in morph_set and base_hyps[i] == morph_hyps[i]]

print(f"Baseline changed: {len(base_changed)}, morph changed: {len(morph_changed)}")
print(f"Only-baseline: {len(only_base)}, only-morph: {len(only_morph)}, both-agree: {len(both)}")

random.shuffle(only_base);  random.shuffle(only_morph); random.shuffle(both)
sel_base  = only_base[:N_EACH]
sel_morph = only_morph[:N_EACH]
sel_both  = both[:N_EACH]

# If any bucket is underfull, fill from remaining changed sentences
all_changed = list(set(base_changed + morph_changed))
random.shuffle(all_changed)
used = set(sel_base + sel_morph + sel_both)
filler = [i for i in all_changed if i not in used]
for lst in [sel_base, sel_morph, sel_both]:
    while len(lst) < N_EACH and filler:
        lst.append(filler.pop(0))

# Build pairs — use baseline hypothesis as the "corrected" text (primary model)
pairs = []
def make_pair(idx, model_tag):
    src = srcs[idx]
    corrected = base_hyps[idx] if model_tag in ("baseline", "both") else morph_hyps[idx]
    pid = hashlib.sha1(f"{src}|{corrected}".encode()).hexdigest()[:12]
    return {
        "pair_id": pid,
        "source": src,
        "corrected": corrected,
        "reference": tgts[idx],
        "model": model_tag,
    }

for i in sel_base:  pairs.append(make_pair(i, "baseline"))
for i in sel_morph: pairs.append(make_pair(i, "morphaware"))
for i in sel_both:  pairs.append(make_pair(i, "both"))

# Shuffle so annotators don't see model grouping
random.shuffle(pairs)
# Strip model tag from annotator-facing pairs (blind evaluation)
blind_pairs = [{"pair_id": p["pair_id"], "source": p["source"], "corrected": p["corrected"]} for p in pairs]

os.makedirs(OUT_DIR, exist_ok=True)
with open(f"{OUT_DIR}/evaluation_pairs.jsonl", "w", encoding="utf-8") as f:
    for p in blind_pairs:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")
# Keep manifest with model labels (for analysis only, not shown to annotators)
with open(f"{OUT_DIR}/evaluation_pairs_manifest.jsonl", "w", encoding="utf-8") as f:
    for p in pairs:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")

print(f"\nWrote {len(blind_pairs)} pairs to {OUT_DIR}/evaluation_pairs.jsonl")
print(f"Manifest (with model tags) -> {OUT_DIR}/evaluation_pairs_manifest.jsonl")
