"""Run the non-neural correction baselines on the natural test set.

Phase 5 (audit items 5.1/5.2/5.4/5.5). The 647-pair synthetic test split lets the
confusion-table baselines (reverse-rule, char trigram LM) exploit the generator's
own CONFUSION_CHARS table. The natural test set was produced by humans, so its
errors do not come from that table -- which is exactly the condition under which
the oracle advantage should disappear. This script measures that.

It also reports the genuinely-natural subset separately: data/natural_test holds
487 rows, but 287 carry source_url "synthetic_seed" (re-seeded from the synthetic
corpus). Only the 200 rows from dissertation/ktc sources are human-authored.

No GPU. Reuses scripts/eval_baselines.py strategies and the same word-level and
span-level scorers used for the neural models.
"""

import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.eval_baselines import (
    CharTrigramLM,
    correct_copy,
    correct_hunspell,
    correct_ngram_lm,
    correct_reverse_rule,
    load_split,
)
from src.evaluation.f05_scorer import evaluate_corpus, evaluate_corpus_span
from src.morphology.lexicon import SoraniLexicon

NAT_PATH = "data/natural_test/sentences.jsonl"
OUT_DIR = "results/baselines"


def load_natural():
    rows = [json.loads(l) for l in open(NAT_PATH, encoding="utf-8")]
    return rows


def subset(rows, prefixes=None):
    if prefixes is None:
        sel = rows
    else:
        sel = [r for r in rows if r["source_url"].split(":")[0] in prefixes]
    srcs = [r["source_text"] for r in sel]
    tgts = [r["target_text"] for r in sel]
    return srcs, tgts, sel


def run_block(name, srcs, tgts, lex, lm):
    strategies = {
        "copy": lambda s: correct_copy(s),
        "hunspell": lambda s: correct_hunspell(s, lex),
        "reverse_rule": lambda s: correct_reverse_rule(s, lex),
        "ngram_lm": lambda s: correct_ngram_lm(s, lex, lm),
    }
    block = {"subset": name, "n_sentences": len(srcs),
             "n_edited": sum(1 for s, t in zip(srcs, tgts) if s != t),
             "systems": []}
    print(f"\n== {name}  (n={len(srcs)}, edited={block['n_edited']}) ==")
    for sysname, fn in strategies.items():
        hyps = fn(srcs)
        mw = evaluate_corpus(srcs, hyps, tgts)
        ms, _ = evaluate_corpus_span(srcs, hyps, tgts)
        rec = {
            "system": sysname,
            "word_f05": round(mw.f05, 4), "word_p": round(mw.precision, 4), "word_r": round(mw.recall, 4),
            "span_f05": round(ms.f05, 4), "span_p": round(ms.precision, 4), "span_r": round(ms.recall, 4),
        }
        block["systems"].append(rec)
        print(f"  {sysname:14s} word F0.5={mw.f05:.4f} (P={mw.precision:.4f} R={mw.recall:.4f})  "
              f"span F0.5={ms.f05:.4f}")
    return block


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = load_natural()
    src_pref = Counter(r["source_url"].split(":")[0] for r in rows)
    print("source prefixes:", dict(src_pref))

    train_srcs, train_tgts = load_split("train")
    lex = SoraniLexicon()
    lm = CharTrigramLM(k=0.1)
    lm.train(train_tgts)
    print(f"Lexicon {len(lex.words)} entries; LM {len(lm.trigram)} trigrams")

    out = {"source_prefix_counts": dict(src_pref), "blocks": []}

    # Full set (mixed; 287 synthetic-seeded).
    s, t, _ = subset(rows)
    out["blocks"].append(run_block("full_487_mixed", s, t, lex, lm))

    # Genuinely natural (human-authored: dissertation + ktc).
    s, t, sel = subset(rows, {"dissertation", "ktc"})
    out["blocks"].append(run_block("natural_200_human", s, t, lex, lm))

    # Synthetic-seeded subset, for contrast (oracle advantage should reappear).
    s, t, _ = subset(rows, {"synthetic_seed"})
    out["blocks"].append(run_block("synthetic_seed_287", s, t, lex, lm))

    # Per-phenomenon breakdown on the human subset (audit 5.5).
    et = Counter(e for r in sel for e in r.get("error_types", []))
    out["natural_human_error_types"] = dict(et)
    print("\nnatural human error_types:", dict(et))

    with open(f"{OUT_DIR}/natural_eval.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nSaved -> {OUT_DIR}/natural_eval.json")


if __name__ == "__main__":
    main()
