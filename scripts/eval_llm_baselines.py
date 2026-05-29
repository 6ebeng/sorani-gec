"""Scaffold for LLM and large fine-tuned comparative baselines.

Reviewer item R5 asks for comparison against stronger systems: instruction
LLMs (GPT-4-class, Aya-23) and seq2seq models that are bigger or multilingual
(mT5, mBART-50). Those need either a paid API key or a GPU larger than what I
had metered access to, so I am not spending on them inside this run. This file
makes each baseline runnable on demand: it guards every backend behind an
explicit flag and a key/availability check, writes hypotheses in the same
JSONL schema as scripts/eval_baselines.py, and scores with the shared F0.5
metric so results drop straight into the comparison table.

Run examples:
    # zero-shot GPT-4-class via OpenAI
    OPENAI_API_KEY=... python scripts/eval_llm_baselines.py --system gpt4

    # Aya-23-8B on a GPU
    python scripts/eval_llm_baselines.py --system aya --device cuda

    # mT5 / mBART-50 (require a fine-tuned checkpoint path)
    python scripts/eval_llm_baselines.py --system mt5 --ckpt path/to/mt5
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.evaluation.f05_scorer import evaluate_corpus

DATA_DIR = "data/splits_v2"
OUT_DIR = "results/baselines"

# A short Sorani correction instruction used for the prompt-based systems.
PROMPT_TEMPLATE = (
    "ئەم ڕستەیە بە کوردیی سۆرانی هەڵەی ڕێزمانی و ڕێنووسی تێدایە. "
    "تکایە تەنها ڕستە ڕاستکراوەکە بنووسەرەوە، بێ هیچ ڕوونکردنەوەیەک.\n\n"
    "ڕستە: {sentence}\nڕاستکراو:"
)


def load_test():
    srcs, tgts = [], []
    with open(f"{DATA_DIR}/test.jsonl", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            srcs.append(rec["source"])
            tgts.append(rec["target"])
    return srcs, tgts


def run_openai(sentences, model="gpt-4o"):
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("OPENAI_API_KEY not set. Aborting without spending.")
    from openai import OpenAI  # lazy import
    client = OpenAI(api_key=key)
    out = []
    for s in sentences:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": PROMPT_TEMPLATE.format(sentence=s)}],
            temperature=0.0,
        )
        out.append(resp.choices[0].message.content.strip())
    return out


def run_hf_causal(sentences, model_name, device="cuda"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    model.eval()
    out = []
    with torch.no_grad():
        for s in sentences:
            prompt = PROMPT_TEMPLATE.format(sentence=s)
            ids = tok(prompt, return_tensors="pt").to(device)
            gen = model.generate(**ids, max_new_tokens=128, do_sample=False)
            text = tok.decode(gen[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
            out.append(text.strip().splitlines()[0] if text.strip() else s)
    return out


def run_hf_seq2seq(sentences, model_name_or_ckpt, device="cuda"):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name_or_ckpt)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_ckpt).to(device)
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(sentences), 16):
            batch = sentences[i:i + 16]
            ids = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
            gen = model.generate(**ids, num_beams=4, max_length=256)
            out.extend(tok.batch_decode(gen, skip_special_tokens=True))
    return out


SYSTEMS = {
    "gpt4":    lambda s, a: run_openai(s, model=a.model or "gpt-4o"),
    "aya":     lambda s, a: run_hf_causal(s, a.model or "CohereForAI/aya-23-8B", a.device),
    "mt5":     lambda s, a: run_hf_seq2seq(s, a.ckpt or "google/mt5-base", a.device),
    "mbart50": lambda s, a: run_hf_seq2seq(s, a.ckpt or "facebook/mbart-large-50", a.device),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True, choices=list(SYSTEMS))
    ap.add_argument("--model", default=None, help="model id override")
    ap.add_argument("--ckpt", default=None, help="fine-tuned checkpoint path (mt5/mbart50)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    srcs, tgts = load_test()
    print(f"Test: {len(srcs)} sentences  | system={args.system}")

    hyps = SYSTEMS[args.system](srcs, args)
    m = evaluate_corpus(srcs, hyps, tgts)
    print(f"  {args.system}: F0.5={m.f05:.4f}  P={m.precision:.4f}  R={m.recall:.4f}")

    with open(f"{OUT_DIR}/{args.system}_hypotheses.jsonl", "w", encoding="utf-8") as f:
        for s, h, t in zip(srcs, hyps, tgts):
            f.write(json.dumps({"source": s, "hypothesis": h, "reference": t}, ensure_ascii=False) + "\n")
    rec = {"system": args.system, "f05": m.f05, "precision": m.precision,
           "recall": m.recall, "tp": m.tp, "fp": m.fp, "fn": m.fn, "n_sentences": len(srcs)}
    with open(f"{OUT_DIR}/{args.system}_summary.json", "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2, ensure_ascii=False)
    print(f"Saved -> {OUT_DIR}/{args.system}_*.{{jsonl,json}}")


if __name__ == "__main__":
    main()
