"""
Phase 3 — R3: Aya-Expanse-8B Zero-Shot GEC Evaluation

Runs the Aya-Expanse-8B model (CohereForAI/aya-expanse-8b) on the
287-pair edited test subset (source != target pairs) and computes
F0.5, GLEU, and CER-floored agreement accuracy.

Usage:
    python scripts/11_evaluate_llm_baseline.py \
        --test-data data/splits/test.jsonl \
        --edited-only \
        --output results/llm_baseline/ \
        --model CohereForAI/aya-expanse-8b \
        [--load-in-4bit]    # use bitsandbytes 4-bit quantization
"""
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.evaluation.f05_scorer import evaluate_corpus
from src.evaluation.agreement_accuracy import evaluate_agreement_accuracy
from src.evaluation.gleu_scorer import compute_gleu

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


SYSTEM_PROMPT = (
    "You are a Kurdish grammar correction assistant. "
    "Correct any grammatical errors in the given Sorani Kurdish sentence. "
    "Return ONLY the corrected sentence, nothing else. "
    "If the sentence is already correct, return it unchanged."
)


def build_prompt(source: str) -> str:
    """Build a zero-shot GEC prompt for Aya."""
    return f"Correct this Sorani Kurdish sentence grammatically:\n\n{source}"


def load_test_data(path: Path, edited_only: bool) -> tuple[list[str], list[str]]:
    sources, references = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            src = rec.get("source", "")
            tgt = rec.get("target", "")
            if edited_only and src == tgt:
                continue
            sources.append(src)
            references.append(tgt)
    logger.info("Loaded %d test pairs (edited_only=%s)", len(sources), edited_only)
    return sources, references


def run_aya_inference(
    sources: list[str],
    model_name: str,
    load_in_4bit: bool,
    batch_size: int,
    max_new_tokens: int,
) -> list[str]:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    logger.info("Loading model: %s (4bit=%s)", model_name, load_in_4bit)

    quant_config = None
    dtype = torch.float16
    if load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        dtype = None  # let BnB handle it

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"  # required for correct generation with decoder-only models
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        torch_dtype=dtype,
        device_map="auto",
    )
    model.eval()
    logger.info("Model loaded.")

    hypotheses = []
    t0 = time.time()

    for i in range(0, len(sources), batch_size):
        batch = sources[i : i + batch_size]
        prompts = [build_prompt(s) for s in batch]

        # Format using chat template if available (includes system message)
        chats = [
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": p}]
            for p in prompts
        ]
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                inputs_text = [
                    tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
                    for chat in chats
                ]
            except Exception:
                # Some models don't support system role — fall back to user-only
                chats_no_sys = [[{"role": "user", "content": SYSTEM_PROMPT + "\n\n" + p}] for p in prompts]
                inputs_text = [
                    tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=True)
                    for c in chats_no_sys
                ]
        else:
            inputs_text = [SYSTEM_PROMPT + "\n\n" + p for p in prompts]

        inputs = tokenizer(
            inputs_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(model.device)

        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode only the generated tokens (not the prompt)
        prompt_len = inputs["input_ids"].shape[1]
        for j, out in enumerate(out_ids):
            generated = out[prompt_len:]
            decoded = tokenizer.decode(generated, skip_special_tokens=True).strip()
            # Fallback: if empty, return source unchanged
            if not decoded:
                decoded = batch[j]
            hypotheses.append(decoded)

        elapsed = time.time() - t0
        logger.info(
            "Batch %d/%d done (%.1fs, %.1f sent/s)",
            min(i + batch_size, len(sources)),
            len(sources),
            elapsed,
            (i + batch_size) / max(elapsed, 0.01),
        )

    return hypotheses


def compute_cer(sources: list[str], hypotheses: list[str]) -> float:
    """Compute average Character Error Rate."""
    import editdistance

    total_cer = 0.0
    for src, hyp in zip(sources, hypotheses):
        if len(src) == 0:
            continue
        cer = editdistance.eval(src, hyp) / len(src)
        total_cer += cer
    return total_cer / max(len(sources), 1)


def main():
    parser = argparse.ArgumentParser(description="Aya-Expanse-8B zero-shot GEC baseline")
    parser.add_argument("--test-data", default="data/splits/test.jsonl")
    parser.add_argument("--edited-only", action="store_true", default=True,
                        help="Evaluate on edited (source != target) pairs only")
    parser.add_argument("--output", default="results/llm_baseline")
    parser.add_argument("--model", default="CohereForAI/aya-expanse-8b",
                        help="HuggingFace model ID")
    parser.add_argument("--load-in-4bit", action="store_true", default=False,
                        help="Use 4-bit quantization (requires bitsandbytes)")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load test data
    test_path = Path(args.test_data)
    sources, references = load_test_data(test_path, args.edited_only)

    if not sources:
        logger.error("No test pairs loaded. Check --test-data path.")
        return

    # Run inference
    hypotheses = run_aya_inference(
        sources=sources,
        model_name=args.model,
        load_in_4bit=args.load_in_4bit,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )

    # Compute metrics
    logger.info("Computing metrics...")
    f05_metrics = evaluate_corpus(sources, hypotheses, references)
    agr = evaluate_agreement_accuracy(hypotheses)
    gleu = compute_gleu(sources, hypotheses, references)

    # CER-floored agreement accuracy
    try:
        avg_cer = compute_cer(sources, hypotheses)
    except ImportError:
        logger.warning("editdistance not installed; CER not computed")
        avg_cer = None

    cer_floor_threshold = 0.5
    cer_floored_agr = agr["accuracy"]
    if avg_cer is not None and avg_cer > cer_floor_threshold:
        cer_floored_agr = 0.0

    results = {
        "model": args.model,
        "load_in_4bit": args.load_in_4bit,
        "test_pairs": len(sources),
        "edited_only": args.edited_only,
        "precision": f05_metrics.precision,
        "recall": f05_metrics.recall,
        "f05": f05_metrics.f05,
        "tp": f05_metrics.tp,
        "fp": f05_metrics.fp,
        "fn": f05_metrics.fn,
        "gleu": gleu,
        "agreement_accuracy_raw": agr["accuracy"],
        "avg_cer": avg_cer,
        "agreement_accuracy_cer_floored": cer_floored_agr,
    }

    # Save results
    results_path = output_dir / "metrics.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Save hypotheses for inspection
    hyps_path = output_dir / "hypotheses.jsonl"
    with open(hyps_path, "w", encoding="utf-8") as f:
        for src, hyp, ref in zip(sources, hypotheses, references):
            f.write(json.dumps({"source": src, "hypothesis": hyp, "reference": ref},
                               ensure_ascii=False) + "\n")

    logger.info("Results saved to %s", results_path)
    print("\n=== Aya-Expanse-8B Zero-Shot Results ===")
    print(f"  Pairs evaluated:   {len(sources)}")
    print(f"  F0.5:              {f05_metrics.f05:.4f}")
    print(f"  Precision:         {f05_metrics.precision:.4f}")
    print(f"  Recall:            {f05_metrics.recall:.4f}")
    print(f"  TP/FP/FN:          {f05_metrics.tp}/{f05_metrics.fp}/{f05_metrics.fn}")
    print(f"  GLEU:              {gleu:.4f}")
    if avg_cer is not None:
        print(f"  Avg CER:           {avg_cer:.4f}")
    print(f"  Agr. Acc. (raw):   {agr['accuracy']:.4f}")
    print(f"  Agr. Acc. (floor): {cer_floored_agr:.4f}")


if __name__ == "__main__":
    main()
