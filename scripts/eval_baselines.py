"""Non-neural correction baselines for Sorani GEC.

Reviewer items R5 and R38: the morphology-aware and baseline ByT5 models need
something to be measured against besides each other. This script runs four
correction strategies that need no GPU, scores each with the same F0.5 metric
used for the neural models, dumps per-sentence hypotheses for the paired
bootstrap, and writes a combined summary.

Strategies
----------
copy
    Return the input unchanged. The do-nothing lower bound; a system that
    cannot beat copy on F0.5 is not editing usefully.
hunspell
    Dictionary spell-correction. Tokenise on whitespace; for every token the
    lexicon rejects, take the first REP-rule suggestion that lands on a real
    word. Pure SoraniLexicon, no external Hunspell binary.
reverse_rule
    Undo the confusion-pair substitutions that the error generators inject
    (l/ll, r/rr, t/T, d/D, z/zh and the orthographic vowel swaps). For each
    rejected token, try every single-character reversal and keep the one that
    yields a valid word.
ngram_lm
    A character trigram language model trained on the clean training
    references. Generate candidates for a rejected token from the lexicon plus
    confusion reversals, then keep the candidate with the best LM log-prob.

Usage:
    python scripts/eval_baselines.py
"""

import json
import math
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.evaluation.f05_scorer import evaluate_corpus
from src.morphology.lexicon import SoraniLexicon

DATA_DIR = "data/splits_v2"
OUT_DIR = "results/baselines"

# Confusion pairs shared with the error generators (spelling_confusion +
# orthography). Each tuple is a symmetric single-character swap we may reverse.
CONFUSION_CHARS: list[tuple[str, str]] = [
    ("ل", "ڵ"), ("ر", "ڕ"), ("ت", "ط"), ("د", "ض"), ("ز", "ژ"),
    ("ح", "ه"), ("غ", "خ"), ("ع", "ئ"), ("ێ", "ی"), ("ۆ", "و"),
    ("ژ", "ز"), ("ڤ", "ف"),
]


def load_split(name: str):
    srcs, tgts = [], []
    with open(f"{DATA_DIR}/{name}.jsonl", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            srcs.append(rec["source"])
            tgts.append(rec["target"])
    return srcs, tgts


# ── Strategy: copy ───────────────────────────────────────────────

def correct_copy(sentences: list[str]) -> list[str]:
    return list(sentences)


# ── Strategy: hunspell (dictionary + REP suggest) ────────────────

def _correct_token_hunspell(tok: str, lex: SoraniLexicon) -> str:
    if not tok or lex.is_valid(tok):
        return tok
    cands = lex.suggest(tok)
    return cands[0] if cands else tok


def correct_hunspell(sentences: list[str], lex: SoraniLexicon) -> list[str]:
    out = []
    for s in sentences:
        toks = s.split()
        out.append(" ".join(_correct_token_hunspell(t, lex) for t in toks))
    return out


# ── Strategy: reverse_rule (undo confusion swaps) ────────────────

def _reverse_candidates(tok: str) -> list[str]:
    """All single-character confusion reversals of a token."""
    cands = []
    for a, b in CONFUSION_CHARS:
        if a in tok:
            cands.append(tok.replace(a, b, 1))
        if b in tok:
            cands.append(tok.replace(b, a, 1))
        # doubled-waw orthography (و / وو)
    if "وو" in tok:
        cands.append(tok.replace("وو", "و", 1))
    elif "و" in tok:
        cands.append(tok.replace("و", "وو", 1))
    return cands


def _correct_token_reverse(tok: str, lex: SoraniLexicon) -> str:
    if not tok or lex.is_valid(tok):
        return tok
    for cand in _reverse_candidates(tok):
        if cand != tok and lex.is_valid(cand):
            return cand
    # fall back to dictionary suggestion
    cands = lex.suggest(tok)
    return cands[0] if cands else tok


def correct_reverse_rule(sentences: list[str], lex: SoraniLexicon) -> list[str]:
    out = []
    for s in sentences:
        toks = s.split()
        out.append(" ".join(_correct_token_reverse(t, lex) for t in toks))
    return out


# ── Strategy: char trigram LM rescoring ──────────────────────────

class CharTrigramLM:
    """Add-k smoothed character trigram model over clean references."""

    def __init__(self, k: float = 0.1):
        self.k = k
        self.bigram = defaultdict(int)
        self.trigram = defaultdict(int)
        self.vocab = set()

    def train(self, texts: list[str]) -> None:
        for t in texts:
            seq = ["<s>", "<s>"] + list(t) + ["</s>"]
            self.vocab.update(seq)
            for i in range(2, len(seq)):
                self.bigram[(seq[i - 2], seq[i - 1])] += 1
                self.trigram[(seq[i - 2], seq[i - 1], seq[i])] += 1

    def logprob(self, text: str) -> float:
        v = max(len(self.vocab), 1)
        seq = ["<s>", "<s>"] + list(text) + ["</s>"]
        lp = 0.0
        for i in range(2, len(seq)):
            ctx = (seq[i - 2], seq[i - 1])
            num = self.trigram.get((seq[i - 2], seq[i - 1], seq[i]), 0) + self.k
            den = self.bigram.get(ctx, 0) + self.k * v
            lp += math.log(num / den)
        return lp


def _correct_token_ngram(tok: str, lex: SoraniLexicon, lm: CharTrigramLM) -> str:
    if not tok or lex.is_valid(tok):
        return tok
    candidates = {tok}
    for c in _reverse_candidates(tok):
        if lex.is_valid(c):
            candidates.add(c)
    for c in lex.suggest(tok):
        candidates.add(c)
    if len(candidates) == 1:
        return tok
    return max(candidates, key=lm.logprob)


def correct_ngram_lm(sentences: list[str], lex: SoraniLexicon, lm: CharTrigramLM) -> list[str]:
    out = []
    for s in sentences:
        toks = s.split()
        out.append(" ".join(_correct_token_ngram(t, lex, lm) for t in toks))
    return out


# ── Driver ───────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    train_srcs, train_tgts = load_split("train")
    test_srcs, test_tgts = load_split("test")
    print(f"Train: {len(train_tgts)}  Test: {len(test_srcs)}")

    lex = SoraniLexicon()
    print(f"Lexicon entries: {len(lex.words)}")

    lm = CharTrigramLM(k=0.1)
    lm.train(train_tgts)
    print(f"LM trained: {len(lm.trigram)} trigrams, vocab {len(lm.vocab)}")

    strategies = {
        "copy":         lambda s: correct_copy(s),
        "hunspell":     lambda s: correct_hunspell(s, lex),
        "reverse_rule": lambda s: correct_reverse_rule(s, lex),
        "ngram_lm":     lambda s: correct_ngram_lm(s, lex, lm),
    }

    summary = []
    for name, fn in strategies.items():
        t0 = time.time()
        hyps = fn(test_srcs)
        elapsed = time.time() - t0
        m = evaluate_corpus(test_srcs, hyps, test_tgts)
        rec = {
            "system": name,
            "f05": m.f05, "precision": m.precision, "recall": m.recall,
            "tp": m.tp, "fp": m.fp, "fn": m.fn,
            "elapsed_sec": round(elapsed, 2), "n_sentences": len(test_srcs),
        }
        summary.append(rec)
        print(f"  {name:14s} F0.5={m.f05:.4f}  P={m.precision:.4f}  R={m.recall:.4f}  ({elapsed:.1f}s)")

        with open(f"{OUT_DIR}/{name}_hypotheses.jsonl", "w", encoding="utf-8") as f:
            for s, h, t in zip(test_srcs, hyps, test_tgts):
                f.write(json.dumps({"source": s, "hypothesis": h, "reference": t}, ensure_ascii=False) + "\n")

    with open(f"{OUT_DIR}/baseline_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSaved -> {OUT_DIR}/baseline_summary.json")


if __name__ == "__main__":
    main()
