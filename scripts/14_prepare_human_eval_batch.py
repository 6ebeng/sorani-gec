"""Prepare a 60-pair human evaluation batch for inter-rater agreement study (R8).

Reads model prediction files from Phase 3 evaluation runs (or falls back to
the test split reference corrections when model hypotheses are unavailable
locally) and produces `results/human_eval/evaluation_pairs.jsonl` consumed by
the Gradio web interface (`web/evaluation.py`).

Selection strategy:
  1. Load source sentences + model predictions from a chosen model directory.
  2. Filter to pairs where the model actually changed something
     (source != hypothesis), so annotators evaluate real corrections.
  3. Stratify by error_type (from test.jsonl) to cover diverse agreement
     categories rather than picking 60 orthography pairs by chance.
  4. Sample up to `--n-pairs` total (default 60), aiming for proportional
     representation of each error type present in the edited subset.

Usage:
    # Use baseline_p3 model predictions (after downloading from remote)
    python scripts/14_prepare_human_eval_batch.py \
        --model-dir results/models/baseline_p3

    # Fallback: use reference corrections from test.jsonl (no model required)
    python scripts/14_prepare_human_eval_batch.py --use-references

    # Override output directory
    python scripts/14_prepare_human_eval_batch.py \
        --model-dir results/models/baseline_p3 \
        --output-dir results/human_eval/baseline_p3
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
TEST_JSONL = ROOT / "data" / "splits" / "test.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "human_eval"


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_test_data(path: Path) -> list[dict]:
    """Load test.jsonl; return list of records."""
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    logger.info("Loaded %d test records from %s", len(records), path)
    return records


def _load_hypotheses(model_dir: Path) -> list[str] | None:
    """Try to load model hypotheses from evaluation output.

    Looks for (in order):
      1. evaluation_pairs.jsonl  (written by 07_evaluate.py; has source+corrected)
      2. hypotheses.txt          (plain one-hypothesis-per-line)
    Returns list of hypothesis strings, or None if not found.
    """
    ep_path = model_dir / "evaluation_pairs.jsonl"
    if ep_path.exists():
        hyps = []
        with ep_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    hyps.append(json.loads(line).get("corrected", ""))
        logger.info("Loaded %d hypotheses from %s", len(hyps), ep_path)
        return hyps

    hyp_path = model_dir / "hypotheses.txt"
    if hyp_path.exists():
        hyps = [ln.rstrip("\n") for ln in hyp_path.read_text("utf-8").splitlines()]
        logger.info("Loaded %d hypotheses from %s", len(hyps), hyp_path)
        return hyps

    logger.warning("No hypothesis file found in %s", model_dir)
    return None


def _primary_error_type(errors: list[dict]) -> str:
    """Return the most specific (first) error type from the errors list."""
    if not errors:
        return "unknown"
    etype = errors[0].get("type", "unknown")
    # Normalise to coarse category for stratification
    if "subject_verb" in etype or "object_verb" in etype:
        return "subject_verb"
    if "clitic" in etype:
        return "clitic"
    if "orthography" in etype or "spelling" in etype:
        return "orthography"
    if "noun_adjective" in etype:
        return "noun_adjective"
    if "word_order" in etype:
        return "word_order"
    if "ezafe" in etype:
        return "ezafe"
    return etype


def _stratified_sample(
    candidates: list[dict],
    n: int,
    rng: random.Random,
) -> list[dict]:
    """Stratify candidates by error_type and sample proportionally up to n."""
    buckets: dict[str, list[dict]] = defaultdict(list)
    for c in candidates:
        buckets[c["_error_type"]].append(c)

    bucket_names = sorted(buckets)
    logger.info("Error-type distribution in candidates: %s",
                {k: len(v) for k, v in sorted(buckets.items(), key=lambda x: -len(x[1]))})

    # Calculate per-bucket quota proportional to bucket size
    total = len(candidates)
    selected: list[dict] = []
    remainder: list[dict] = []

    for bname in bucket_names:
        items = buckets[bname]
        quota = max(1, round(len(items) / total * n))
        rng.shuffle(items)
        selected.extend(items[:quota])
        remainder.extend(items[quota:])

    # Fill up to n from remainder if under
    if len(selected) < n:
        rng.shuffle(remainder)
        selected.extend(remainder[: n - len(selected)])

    # Trim to n if slightly over due to rounding
    rng.shuffle(selected)
    return selected[:n]


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare evaluation_pairs.jsonl for the human evaluation web app.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Directory containing model evaluation outputs (hypotheses.txt or evaluation_pairs.jsonl).",
    )
    parser.add_argument(
        "--use-references",
        action="store_true",
        help="Use reference corrections from test.jsonl instead of model hypotheses. "
             "Useful when model outputs are not yet downloaded from remote.",
    )
    parser.add_argument(
        "--test-data",
        type=Path,
        default=TEST_JSONL,
        help="Path to test.jsonl (default: data/splits/test.jsonl).",
    )
    parser.add_argument(
        "--n-pairs",
        type=int,
        default=60,
        help="Number of evaluation pairs to output (default: 60).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible stratified sampling.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write evaluation_pairs.jsonl (default: results/human_eval/).",
    )
    args = parser.parse_args()

    if not args.use_references and args.model_dir is None:
        parser.error("Provide --model-dir or --use-references.")

    if not args.test_data.exists():
        raise SystemExit(f"Test data not found: {args.test_data}")

    test_records = _load_test_data(args.test_data)
    rng = random.Random(args.seed)

    # Build source → hypothesis pairs
    if args.use_references:
        logger.info("Using reference corrections from test.jsonl (--use-references mode).")
        sources = [r.get("source", r.get("corrupted", "")) for r in test_records]
        hypotheses = [r.get("target", "") for r in test_records]
    else:
        hypotheses = _load_hypotheses(args.model_dir)  # type: ignore[arg-type]
        if hypotheses is None:
            raise SystemExit(
                f"No hypothesis file found in {args.model_dir}. "
                "Download results from remote first, or use --use-references."
            )
        sources = [r.get("source", r.get("corrupted", "")) for r in test_records]
        if len(hypotheses) != len(sources):
            raise SystemExit(
                f"Hypothesis count ({len(hypotheses)}) != test record count ({len(sources)}). "
                "Check that the hypothesis file matches the test split."
            )

    # Build candidate list: only pairs where model changed something
    candidates: list[dict] = []
    for record, src, hyp in zip(test_records, sources, hypotheses):
        if not src or not hyp:
            continue
        if src.strip() == hyp.strip():
            continue  # Model made no change; not useful for human evaluation
        errors = record.get("errors", [])
        candidates.append(
            {
                "source": src,
                "corrected": hyp,
                "reference": record.get("target", ""),
                "_error_type": _primary_error_type(errors),
                "_category": record.get("category", ""),
            }
        )

    logger.info("Candidates (model changed something): %d / %d", len(candidates), len(test_records))

    if len(candidates) < args.n_pairs:
        logger.warning(
            "Only %d candidates but requested %d pairs; using all candidates.",
            len(candidates), args.n_pairs,
        )
        selected = candidates
        rng.shuffle(selected)
    else:
        selected = _stratified_sample(candidates, args.n_pairs, rng)

    # Write output
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "evaluation_pairs.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for item in selected:
            # Drop private keys before writing
            record_out = {k: v for k, v in item.items() if not k.startswith("_")}
            f.write(json.dumps(record_out, ensure_ascii=False) + "\n")

    logger.info("Wrote %d evaluation pairs to %s", len(selected), out_path)
    logger.info("")
    logger.info("Next steps for R8 human evaluation:")
    logger.info("  1. Start the web app:")
    logger.info("       cd Implementation && python -m web.app --eval-dir %s", args.output_dir)
    logger.info("  2. Share the Gradio link with two native Sorani speakers.")
    logger.info("  3. Each rater selects a unique rater ID and rates all %d pairs.", len(selected))
    logger.info("  4. Once both raters finish, compute kappa:")
    logger.info("       python -c \"")
    logger.info("         from src.evaluation.inter_rater import compute_inter_rater_agreement")
    logger.info("         from pathlib import Path")
    logger.info("         r = compute_inter_rater_agreement(Path('%s'))", args.output_dir)
    logger.info("         print(r)\"")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
