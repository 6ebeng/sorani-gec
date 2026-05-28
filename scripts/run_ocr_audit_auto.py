"""
Automated OCR quality audit for the dissertation corpus (R16).

Methodology
-----------
Each dissertation was digitised from a scanned PDF using Google Gemini's
OCR pipeline.  Measuring OCR quality against a manually transcribed
ground truth requires a human reader — impractical across 100 files.

Instead, this script uses the project's SoraniNormalizer as a canonical
reference: it applies the same normalization rules that the downstream
data pipeline runs on every sentence, then measures how far the raw OCR
text deviates from that canonical form.  The CER/WER figures therefore
quantify *encoding-level noise that propagates into the training corpus*,
which is the operationally important quantity for downstream NLP.

Specifically, the normalization captures:
  - Arabic kaf (U+0643) vs Kurdish kaf (U+06A9)
  - Arabic yeh (U+064A) vs Sorani yeh (U+06CC)
  - Non-initial Arabic heh (U+0647) vs Sorani small heh (U+06D5)
  - Teh Marbuta (U+0629) → U+06D5
  - Western/Arabic-Indic digits vs Extended Arabic-Indic
  - Tatweel (kashida) removal
  - Spurious zero-width characters (ZWJ, ZWS, LRM, RLM, BOM)

Limitation: this methodology measures deviations from a Sorani Unicode
standard, not transcription accuracy against the original PDF.  It
therefore gives a lower bound on the true CER; punctuation or word-
boundary artefacts may not be fully captured.  This limitation is
documented in Chapter 6.

Output
------
  results/ocr_audit/per_university_cer.csv
  results/ocr_audit/ocr_audit.json
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Make src/ importable regardless of working directory
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.data.normalizer import SoraniNormalizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# University detection
# ---------------------------------------------------------------------------

_UNI_PATTERNS: dict[str, re.Pattern[str]] = {
    "koya": re.compile(r"koya", re.IGNORECASE),
    "salahadin": re.compile(r"salahadin", re.IGNORECASE),
    "sulaimanyah": re.compile(r"sulaimanyah", re.IGNORECASE),
}


def _detect_university(path: Path) -> str | None:
    for uni, pat in _UNI_PATTERNS.items():
        if pat.search(path.name) or pat.search(str(path.parent)):
            return uni
    return None


# ---------------------------------------------------------------------------
# Edit distance
# ---------------------------------------------------------------------------

def _edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            tmp = dp[j]
            dp[j] = prev if a[i - 1] == b[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = tmp
    return dp[n]


def cer(hyp: str, ref: str) -> float:
    if not ref:
        return 0.0
    return _edit_distance(hyp, ref) / len(ref)


def wer(hyp: list[str], ref: list[str]) -> float:
    if not ref:
        return 0.0
    return _edit_distance(hyp, ref) / len(ref)


# ---------------------------------------------------------------------------
# Word tokenizer (whitespace split, strip punctuation)
# ---------------------------------------------------------------------------

_PUNCT_STRIP = re.compile(r"[^\u0600-\u06FF\u0750-\u077F\u200C\u200D]+")


def _tokenize(text: str) -> list[str]:
    """Split on whitespace; drop empty and punctuation-only tokens."""
    tokens = []
    for tok in text.split():
        tok = _PUNCT_STRIP.sub("", tok).strip()
        if tok:
            tokens.append(tok)
    return tokens


# ---------------------------------------------------------------------------
# Core audit
# ---------------------------------------------------------------------------

def audit_files(
    ocr_dir: Path,
    sample_n: int,
    seed: int,
) -> dict:
    """Sample words from OCR files; compare against normalized form."""
    normalizer = SoraniNormalizer(
        normalize_chars=True,
        remove_diacritics=False,
        remove_zero_width=True,
        preserve_zwnj=True,
        normalize_whitespace=True,
    )

    # Collect all .txt files, group by university
    uni_files: dict[str, list[Path]] = defaultdict(list)
    txt_files = list(ocr_dir.glob("**/*.txt"))
    logger.info("Found %d .txt files in %s", len(txt_files), ocr_dir)

    for f in txt_files:
        if f.name.startswith("_"):
            continue  # skip _index.json etc.
        uni = _detect_university(f)
        if uni:
            uni_files[uni].append(f)

    if not uni_files:
        raise ValueError(
            f"No university-tagged .txt files found under {ocr_dir}. "
            "Expected filenames containing 'koya', 'salahadin', or 'sulaimanyah'."
        )

    rng = random.Random(seed)
    results: dict[str, dict] = {}

    for uni in sorted(uni_files):
        files = sorted(uni_files[uni])
        logger.info("University %s: %d dissertation files", uni, len(files))

        # Collect all words from all files for this university
        all_words_raw: list[str] = []
        for f in files:
            try:
                text = f.read_text(encoding="utf-8", errors="replace")
            except Exception as exc:
                logger.warning("Could not read %s: %s", f, exc)
                continue
            all_words_raw.extend(text.split())

        if not all_words_raw:
            logger.warning("No words found for university: %s", uni)
            continue

        # Sample words
        n = min(sample_n, len(all_words_raw))
        sampled_raw = rng.sample(all_words_raw, n)

        # Generate normalized reference
        sampled_norm = [normalizer.normalize(w) for w in sampled_raw]

        # Per-word CER
        word_cers = [
            cer(raw, norm) for raw, norm in zip(sampled_raw, sampled_norm)
        ]
        # Corpus-level CER: all chars concatenated
        hyp_concat = "".join(sampled_raw)
        ref_concat = "".join(sampled_norm)
        corpus_cer = cer(hyp_concat, ref_concat) if ref_concat else 0.0

        # Corpus-level WER
        hyp_words = [_tokenize(w) for w in sampled_raw]
        ref_words = [_tokenize(w) for w in sampled_norm]
        hyp_flat = [t for toks in hyp_words for t in toks]
        ref_flat = [t for toks in ref_words for t in toks]
        corpus_wer = wer(hyp_flat, ref_flat) if ref_flat else 0.0

        changed = sum(1 for r, n_ in zip(sampled_raw, sampled_norm) if r != n_)
        change_rate = changed / n if n else 0.0

        results[uni] = {
            "n_files": len(files),
            "n_words_sampled": n,
            "n_words_changed_by_norm": changed,
            "word_change_rate": round(change_rate, 4),
            "mean_word_cer": round(sum(word_cers) / len(word_cers), 4) if word_cers else 0.0,
            "corpus_cer": round(corpus_cer, 4),
            "corpus_wer": round(corpus_wer, 4),
        }
        logger.info(
            "  %s: corpus_cer=%.4f corpus_wer=%.4f changed=%d/%d (%.1f%%)",
            uni,
            corpus_cer,
            corpus_wer,
            changed,
            n,
            change_rate * 100,
        )

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Auto OCR audit using SoraniNormalizer as ground truth (R16)"
    )
    parser.add_argument(
        "--ocr-dir",
        default="kurdish_books_and_dissertations/ocr_output_gemini",
        help="Directory containing university-tagged OCR .txt files",
    )
    parser.add_argument(
        "--out-dir",
        default="results/ocr_audit",
        help="Output directory for CSV and JSON results",
    )
    parser.add_argument("--n", type=int, default=300, help="Words to sample per university")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args(argv)

    # Resolve paths relative to repo root
    ocr_dir = (_REPO_ROOT / args.ocr_dir).resolve()
    # ocr_dir lives outside the sorani-gec tree; also accept absolute paths
    if not ocr_dir.exists():
        # try relative to CWD
        ocr_dir = Path(args.ocr_dir).resolve()
    if not ocr_dir.exists():
        parser.error(f"OCR directory not found: {ocr_dir}")

    out_dir = (_REPO_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    results = audit_files(ocr_dir, args.n, args.seed)

    # CSV output
    csv_path = out_dir / "per_university_cer.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "university",
                "n_files",
                "n_words_sampled",
                "n_words_changed_by_norm",
                "word_change_rate",
                "mean_word_cer",
                "corpus_cer",
                "corpus_wer",
            ],
        )
        writer.writeheader()
        for uni, row in sorted(results.items()):
            writer.writerow({"university": uni, **row})
    logger.info("CSV written → %s", csv_path)

    # Aggregate stats
    if results:
        agg_cer = sum(r["corpus_cer"] for r in results.values()) / len(results)
        agg_wer = sum(r["corpus_wer"] for r in results.values()) / len(results)
    else:
        agg_cer = agg_wer = 0.0

    audit_doc = {
        "methodology": (
            "Normalization-proxy audit: raw OCR tokens compared against "
            "SoraniNormalizer(normalize_chars=True, remove_zero_width=True, "
            "preserve_zwnj=True) output.  Measures encoding-level deviations "
            "that propagate to the training corpus.  Lower bound on true OCR CER."
        ),
        "seed": args.seed,
        "n_per_university": args.n,
        "per_university": results,
        "aggregate": {
            "n_universities": len(results),
            "mean_corpus_cer": round(agg_cer, 4),
            "mean_corpus_wer": round(agg_wer, 4),
        },
    }

    json_path = out_dir / "ocr_audit.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(audit_doc, f, ensure_ascii=False, indent=2)
    logger.info("JSON written → %s", json_path)

    # Summary to stdout
    print("\n=== OCR Audit Summary (normalization proxy) ===")
    print(f"{'University':<15} {'Files':>5} {'Sample':>6} {'Changed%':>9} {'CER':>7} {'WER':>7}")
    print("-" * 55)
    for uni, row in sorted(results.items()):
        print(
            f"{uni:<15} {row['n_files']:>5} {row['n_words_sampled']:>6} "
            f"{row['word_change_rate']*100:>8.1f}% {row['corpus_cer']:>7.4f} "
            f"{row['corpus_wer']:>7.4f}"
        )
    print("-" * 55)
    print(f"{'Aggregate mean':<15} {'':>5} {'':>6} {'':>9} {agg_cer:>7.4f} {agg_wer:>7.4f}")


if __name__ == "__main__":
    main()
