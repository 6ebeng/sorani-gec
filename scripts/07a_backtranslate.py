"""
R7 — Back-translation corpus expansion.

Back-translates monolingual Central Kurdish (Sorani) text (from KTC or dissertation sentences)
into English via NLLB-200 or Aya-23, then translates the English back to Sorani,
producing (corrupted-back-translated, original) pairs that augment the synthetic
training corpus to ≥ 30,000 pairs.

Methodology (see Chapter 6 §6.2 for prose description):
  1. Load clean Sorani source sentences from ``data/clean/`` or KTC .txt files.
  2. Sorani (ckb_Arab) → English (eng_Latn) via NLLB-200 distilled 600M.
  3. English → Sorani (ckb_Arab) via the reverse NLLB-200 direction.
  4. The round-trip output is the ``source'' (potentially corrupted); the original
     Sorani sentence is the ``target''.
  5. Pairs where source == target (identity round-trips) are kept as copy-through
     signal (consistent with splits_v2 design).
  6. Pairs are deduplicated against splits_v2 at Jaccard ≥ 0.90 (character trigrams)
     to prevent test-set leakage.
  7. Output is written to ``data/splits_v3/`` with SHA-256 manifest.

Requirements:
  pip install transformers sentencepiece sacremoses torch

Usage:
  python scripts/07a_backtranslate.py \\
      --source-dir data/clean \\
      --splits-v2-dir data/splits_v2 \\
      --out-dir data/backtranslated \\
      --target-n 25000 \\
      --batch-size 32 \\
      --seed 42 \\
      --model facebook/nllb-200-distilled-600M

Notes:
  - ``--model aya`` uses CohereForAI/aya-23-8B (requires ~16 GB VRAM).
  - For ByT5 training the back-translated pairs should be merged with splits_v2
    to produce splits_v3; run ``scripts/create_splits_v2.py`` on the merged set.
  - GPU strongly recommended; NLLB 600M runs at ~120 pairs/minute on RTX 3080.
"""

import argparse
import hashlib
import json
import logging
import os
import random
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

NLLB_SRC_LANG = "ckb_Arab"   # Central Kurdish (Sorani) in Arabic script
NLLB_TGT_LANG = "eng_Latn"   # English (for pivot)
NLLB_DEFAULT  = "facebook/nllb-200-distilled-600M"


def _load_source_sentences(source_dir: Path) -> list[str]:
    """Load all .txt files from source_dir; return one sentence per non-empty line."""
    sentences: list[str] = []
    for txt in sorted(source_dir.glob("*.txt")):
        with open(txt, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line and len(line.split()) >= 3:  # skip very short fragments
                    sentences.append(line)
    logger.info("Loaded %d source sentences from %s", len(sentences), source_dir)
    return sentences


def _load_splits_v2_sources(splits_dir: Path) -> set[str]:
    """Return the set of all source sentences in splits_v2 (for dedup guard)."""
    sources: set[str] = set()
    for split in ("train", "dev", "test"):
        for ext in ("jsonl",):
            p = splits_dir / f"{split}.{ext}"
            if p.exists():
                with open(p, "r", encoding="utf-8") as fh:
                    for line in fh:
                        rec = json.loads(line.strip())
                        sources.add(rec.get("source", ""))
                        sources.add(rec.get("target", ""))
    logger.info("Loaded %d existing sentences from splits_v2 for dedup guard", len(sources))
    return sources


def _jaccard_90(a: str, b: str) -> bool:
    """Return True if character-trigram Jaccard(a, b) ≥ 0.90."""
    def trigrams(s: str) -> set[str]:
        return {s[i:i+3] for i in range(len(s) - 2)} if len(s) >= 3 else {s}
    ta, tb = trigrams(a), trigrams(b)
    if not ta or not tb:
        return a == b
    return len(ta & tb) / len(ta | tb) >= 0.90


def _batch_translate(
    model,
    tokenizer,
    sentences: list[str],
    src_lang: str,
    tgt_lang: str,
    batch_size: int,
    device: str,
) -> list[str]:
    """Translate *sentences* from src_lang to tgt_lang in batches."""
    import torch

    tokenizer.src_lang = src_lang
    results: list[str] = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        ).to(device)
        forced_bos = tokenizer.convert_tokens_to_ids(tgt_lang)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                forced_bos_token_id=forced_bos,
                max_new_tokens=256,
                num_beams=4,
            )
        decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        results.extend(decoded)
        if (i // batch_size) % 10 == 0:
            logger.info("  Translated %d / %d sentences", min(i + batch_size, len(sentences)), len(sentences))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Back-translation corpus expansion (R7)")
    parser.add_argument("--source-dir", default="data/clean",
                        help="Directory of clean Sorani .txt files")
    parser.add_argument("--splits-v2-dir", default="data/splits_v2",
                        help="splits_v2 directory (for dedup guard)")
    parser.add_argument("--out-dir", default="data/backtranslated",
                        help="Output directory for back-translated JSONL pairs")
    parser.add_argument("--target-n", type=int, default=25000,
                        help="Target number of back-translated pairs to generate")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default=NLLB_DEFAULT,
                        help="HuggingFace model ID for translation "
                             "(default: facebook/nllb-200-distilled-600M)")
    parser.add_argument("--device", default=None,
                        help="Device override (e.g. 'cuda:0', 'cpu')")
    args = parser.parse_args()

    random.seed(args.seed)

    # ---- lazy imports (GPU environment only) ----
    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError as exc:
        logger.error("transformers not installed: %s", exc)
        logger.error("Run: pip install transformers sentencepiece sacremoses torch")
        sys.exit(1)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # ---- load model ----
    logger.info("Loading translation model: %s", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(device)
    model.eval()

    # ---- load sentences ----
    source_dir = Path(args.source_dir)
    sentences = _load_source_sentences(source_dir)
    if not sentences:
        logger.error("No source sentences found in %s", source_dir)
        sys.exit(1)

    random.shuffle(sentences)
    pool = sentences[: min(args.target_n * 2, len(sentences))]  # 2× safety margin
    logger.info("Using %d candidate sentences", len(pool))

    # ---- load existing splits for dedup ----
    splits_dir = Path(args.splits_v2_dir)
    existing = _load_splits_v2_sources(splits_dir) if splits_dir.exists() else set()

    # ---- round-trip translation ----
    logger.info("Step 1: Sorani → English via %s", args.model)
    english = _batch_translate(model, tokenizer, pool, NLLB_SRC_LANG, NLLB_TGT_LANG,
                                args.batch_size, device)

    logger.info("Step 2: English → Sorani (back-translation)")
    back_sorani = _batch_translate(model, tokenizer, english, NLLB_TGT_LANG, NLLB_SRC_LANG,
                                   args.batch_size, device)

    # ---- build pairs with dedup ----
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "backtranslated_pairs.jsonl"

    kept = 0
    skipped_dedup = 0
    with open(out_path, "w", encoding="utf-8") as fh:
        for original, back in zip(pool, back_sorani):
            if kept >= args.target_n:
                break
            # dedup guard: reject pairs near-duplicate to existing splits
            too_close = any(_jaccard_90(original, ex) for ex in existing)
            if too_close:
                skipped_dedup += 1
                continue
            record = {
                "source": back,        # potentially degraded round-trip output
                "target": original,    # gold original Sorani sentence
                "error_type": "back_translation",
                "provenance": "nllb_round_trip",
                "model": args.model,
                "seed": args.seed,
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1

    logger.info("Wrote %d back-translated pairs to %s", kept, out_path)
    logger.info("Skipped %d pairs (Jaccard-0.90 dedup against splits_v2)", skipped_dedup)

    # ---- SHA-256 manifest entry ----
    with open(out_path, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    manifest = {
        "file": str(out_path),
        "sha256": digest,
        "pairs": kept,
        "model": args.model,
        "seed": args.seed,
        "note": (
            "Back-translated pairs via NLLB-200 Sorani→English→Sorani round-trip. "
            "Merge with splits_v2 to produce splits_v3 (≥30k pairs for R7). "
            "Run scripts/create_splits_v2.py --extra-jsonl <this file> to merge."
        ),
    }
    manifest_path = out_dir / "backtranslate_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    logger.info("Manifest written to %s", manifest_path)
    logger.info("SHA-256: %s", digest)


if __name__ == "__main__":
    main()
