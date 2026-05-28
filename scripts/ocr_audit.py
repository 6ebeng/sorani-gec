"""
OCR quality audit for the dissertation corpus (R16).

Methodology:
  1. Sample N words (default 100) from each university's OCR output.
  2. Compare against manually provided ground-truth transcriptions.
  3. Compute CER = edit_distance(hyp, ref) / len(ref) and
         WER = edit_distance_words(hyp, ref) / len(ref_words).
  4. Report per-university and aggregate statistics.

Usage:
    # Step 1 — generate sample word lists for manual annotation:
    python scripts/ocr_audit.py --mode sample \
        --ocr-dir data/raw \
        --out-dir results/ocr_audit_samples \
        --n 100 --seed 42

    # Step 2 — after manually creating ground-truth files, compute metrics:
    python scripts/ocr_audit.py --mode evaluate \
        --samples-dir results/ocr_audit_samples \
        --out results/ocr_audit.json

Ground-truth format:
    For each <uni>_sample.txt produced in step 1, create <uni>_groundtruth.txt
    in the same directory. Each line: one corrected word, in the same order as
    the sample file.  Lines starting with '#' are treated as comments and
    excluded from alignment.

Universities expected:
    koya, sulaimani, salahaddin  (matched by lowercase prefix of filename)
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Edit distance (Levenshtein)
# ---------------------------------------------------------------------------

def edit_distance(a: str, b: str) -> int:
    """Character-level Levenshtein distance."""
    if a == b:
        return 0
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            tmp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = tmp
    return dp[n]


def cer(hypothesis: str, reference: str) -> float:
    """Character Error Rate = edit_distance / len(reference)."""
    if not reference:
        return 0.0
    return edit_distance(hypothesis, reference) / len(reference)


def wer(hypothesis: list[str], reference: list[str]) -> float:
    """Word Error Rate = edit_distance_words / len(reference_words)."""
    if not reference:
        return 0.0
    return edit_distance(" ".join(hypothesis), " ".join(reference)) / len(reference)


# ---------------------------------------------------------------------------
# Sampling mode
# ---------------------------------------------------------------------------

UNIVERSITY_PREFIXES = {
    "koya":       ["koya", "kok", "كۆيە"],
    "sulaimani":  ["sulaimani", "slemani", "slm", "سلێمانی"],
    "salahaddin": ["salahaddin", "hawler", "hwl", "سەلاحەدین"],
}


def _infer_university(filename: str) -> str:
    lower = filename.lower()
    for uni, prefixes in UNIVERSITY_PREFIXES.items():
        if any(lower.startswith(p) or p in lower for p in prefixes):
            return uni
    return "unknown"


def sample_mode(ocr_dir: Path, out_dir: Path, n: int, seed: int) -> None:
    """Extract N random words from each university's OCR files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    # Group txt files by university
    from collections import defaultdict
    groups: dict[str, list[Path]] = defaultdict(list)
    for f in sorted(ocr_dir.rglob("*.txt")):
        uni = _infer_university(f.stem)
        groups[uni].append(f)

    if not groups:
        logger.error("No .txt files found under %s", ocr_dir)
        sys.exit(1)

    for uni, files in sorted(groups.items()):
        words: list[tuple[str, str]] = []  # (word, source_file)
        for f in files:
            try:
                text = f.read_text(encoding="utf-8", errors="replace")
            except Exception as exc:
                logger.warning("Could not read %s: %s", f, exc)
                continue
            for word in text.split():
                if word.strip():
                    words.append((word.strip(), f.name))

        if not words:
            logger.warning("No words found for university '%s'", uni)
            continue

        sampled = rng.sample(words, min(n, len(words)))
        sample_path = out_dir / f"{uni}_sample.txt"
        gt_placeholder = out_dir / f"{uni}_groundtruth.txt"

        with open(sample_path, "w", encoding="utf-8") as fout:
            fout.write(f"# OCR sample — {uni} — {len(sampled)} words (seed={seed})\n")
            fout.write("# Column: OCR_word  [TAB]  source_file\n")
            for word, src in sampled:
                fout.write(f"{word}\t{src}\n")

        if not gt_placeholder.exists():
            with open(gt_placeholder, "w", encoding="utf-8") as fgt:
                fgt.write(f"# Ground-truth transcriptions for {uni}\n")
                fgt.write("# One corrected word per line, in the same order as the sample file.\n")
                fgt.write("# Lines starting with '#' are skipped.\n")
                for word, _ in sampled:
                    fgt.write(f"{word}\n")  # Prefill with OCR word; annotator corrects

        logger.info(
            "University '%s': sampled %d words from %d files -> %s",
            uni, len(sampled), len(files), sample_path,
        )

    logger.info("Samples written to %s. Edit *_groundtruth.txt files before running --mode evaluate.", out_dir)


# ---------------------------------------------------------------------------
# Evaluation mode
# ---------------------------------------------------------------------------

def evaluate_mode(samples_dir: Path, out_path: Path) -> None:
    """Compute CER/WER from sample + ground-truth file pairs."""
    results: dict = {}

    for sample_file in sorted(samples_dir.glob("*_sample.txt")):
        uni = sample_file.stem.replace("_sample", "")
        gt_file = samples_dir / f"{uni}_groundtruth.txt"

        if not gt_file.exists():
            logger.warning("Ground-truth file missing for '%s': %s", uni, gt_file)
            continue

        def _read_words(path: Path) -> list[str]:
            words = []
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.startswith("#") or not line.strip():
                    continue
                # Sample file has word\tsource; GT file has just word
                words.append(line.split("\t")[0].strip())
            return words

        ocr_words = _read_words(sample_file)
        gt_words  = _read_words(gt_file)

        if len(ocr_words) != len(gt_words):
            logger.warning(
                "'%s': sample has %d words, ground truth has %d. Truncating to shorter.",
                uni, len(ocr_words), len(gt_words),
            )
            n = min(len(ocr_words), len(gt_words))
            ocr_words = ocr_words[:n]
            gt_words  = gt_words[:n]

        if not gt_words:
            logger.warning("No aligned words for '%s'; skipping.", uni)
            continue

        # Per-word CER then aggregate
        per_word_cer = [cer(h, r) for h, r in zip(ocr_words, gt_words)]
        avg_cer      = sum(per_word_cer) / len(per_word_cer)
        # WER on the flat list treated as one sequence
        word_err     = sum(1 for h, r in zip(ocr_words, gt_words) if h != r) / len(gt_words)

        results[uni] = {
            "n_words":    len(gt_words),
            "cer_avg":    round(avg_cer, 4),
            "cer_pct":    round(avg_cer * 100, 2),
            "wer":        round(word_err, 4),
            "wer_pct":    round(word_err * 100, 2),
            "n_errors":   sum(1 for h, r in zip(ocr_words, gt_words) if h != r),
        }
        logger.info(
            "%s: CER=%.1f%%  WER=%.1f%%  (%d/%d words incorrect)",
            uni, avg_cer * 100, word_err * 100,
            results[uni]["n_errors"], len(gt_words),
        )

    if not results:
        logger.error("No results computed. Ensure *_groundtruth.txt files are populated.")
        sys.exit(1)

    # Aggregate across universities
    all_n     = sum(v["n_words"] for v in results.values())
    macro_cer = sum(v["cer_avg"] for v in results.values()) / len(results)
    macro_wer = sum(v["wer"]     for v in results.values()) / len(results)
    results["_aggregate"] = {
        "universities": list(results.keys()),
        "total_words":  all_n,
        "macro_cer_avg": round(macro_cer, 4),
        "macro_cer_pct": round(macro_cer * 100, 2),
        "macro_wer_avg": round(macro_wer, 4),
        "macro_wer_pct": round(macro_wer * 100, 2),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info("OCR audit results written to %s", out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="OCR quality audit for dissertation corpus (R16)")
    ap.add_argument("--mode", choices=["sample", "evaluate"], required=True)

    sample_grp = ap.add_argument_group("sample mode")
    sample_grp.add_argument("--ocr-dir",   default="data/raw",                  help="Directory containing OCR .txt files")
    sample_grp.add_argument("--out-dir",   default="results/ocr_audit_samples", help="Output directory for sample files")
    sample_grp.add_argument("--n",         type=int, default=100,               help="Words to sample per university")
    sample_grp.add_argument("--seed",      type=int, default=42,                help="RNG seed for reproducibility")

    eval_grp = ap.add_argument_group("evaluate mode")
    eval_grp.add_argument("--samples-dir", default="results/ocr_audit_samples", help="Directory with *_sample.txt and *_groundtruth.txt")
    eval_grp.add_argument("--out",         default="results/ocr_audit.json",    help="Output JSON path")

    args = ap.parse_args()

    if args.mode == "sample":
        sample_mode(Path(args.ocr_dir), Path(args.out_dir), args.n, args.seed)
    else:
        evaluate_mode(Path(args.samples_dir), Path(args.out))


if __name__ == "__main__":
    main()
