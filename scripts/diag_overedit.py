"""Quick diagnostic: why does the scaled-data baseline over-edit?

Loads a trained baseline checkpoint, runs it on the first N dev sentences, and
prints source / hypothesis / reference with the word-level edit count the
F0.5 scorer would see. Tells us whether the FP floor is (a) genuine rewriting,
(b) invisible Unicode/normalization churn, or (c) length blow-ups.
"""

import argparse
import json
import os
import sys
import unicodedata

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from src.model.baseline import BaselineGEC
from src.evaluation.f05_scorer import sentence_level_edits


def load_dev(data_dir: str, n: int):
    srcs, tgts = [], []
    with open(f"{data_dir}/dev.jsonl", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            srcs.append(rec["source"])
            tgts.append(rec["target"])
    return srcs[:n], tgts[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", default="data/splits_scaled")
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--beams", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=256)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    srcs, tgts = load_dev(args.data_dir, args.n)

    model = BaselineGEC(model_name="google/byt5-small", max_length=args.max_length)
    state = torch.load(args.ckpt, map_location=device, weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()

    hyps = []
    with torch.no_grad():
        for i in range(0, len(srcs), 16):
            hyps.extend(model.correct_batch(srcs[i:i + 16], num_beams=args.beams))

    tot_hyp_edits = tot_ref_edits = 0
    tot_norm_only = 0          # edits that vanish under NFC normalization
    tot_len_delta = 0
    for s, h, t in zip(srcs, hyps, tgts):
        hyp_edits = sentence_level_edits(s, h)
        ref_edits = sentence_level_edits(s, t)
        # how many hyp edits are pure normalization (same string after NFC)?
        norm_only = sum(
            1 for a, b in hyp_edits
            if unicodedata.normalize("NFC", a) == unicodedata.normalize("NFC", b)
        )
        tot_hyp_edits += len(hyp_edits)
        tot_ref_edits += len(ref_edits)
        tot_norm_only += norm_only
        tot_len_delta += len(h.split()) - len(s.split())

    n = len(srcs)
    print("=" * 72)
    print(f"checkpoint: {args.ckpt}")
    print(f"dev sentences inspected: {n}")
    print(f"avg hyp edits/sent (model vs source): {tot_hyp_edits / n:.2f}")
    print(f"avg ref edits/sent (gold vs source):  {tot_ref_edits / n:.2f}")
    print(f"hyp edits that are NFC-normalization-only: {tot_norm_only} "
          f"({100 * tot_norm_only / max(tot_hyp_edits, 1):.1f}% of hyp edits)")
    print(f"avg word-length delta (hyp - source): {tot_len_delta / n:+.2f}")
    print("=" * 72)

    for idx, (s, h, t) in enumerate(zip(srcs, hyps, tgts)):
        if idx >= 12:
            break
        print(f"\n--- dev[{idx}] ---")
        print(f"SRC: {s}")
        print(f"HYP: {h}")
        print(f"REF: {t}")
        print(f"  hyp_edits={len(sentence_level_edits(s, h))}  "
              f"ref_edits={len(sentence_level_edits(s, t))}  "
              f"src_words={len(s.split())} hyp_words={len(h.split())}")


if __name__ == "__main__":
    main()
