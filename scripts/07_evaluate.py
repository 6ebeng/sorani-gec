"""
Step 7: Evaluate GEC Models

Supports both baseline (BaselineGEC) and morphology-aware (MorphologyAwareGEC)
model evaluation on the test split.

Usage:
    python scripts/07_evaluate.py --model-path results/models/baseline/best_model.pt
    python scripts/07_evaluate.py --model-path results/models/morphaware/best_model.pt --morphaware
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.evaluation.f05_scorer import evaluate_corpus
from src.evaluation.agreement_accuracy import evaluate_agreement_accuracy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_test_data(test_src: str, test_ref: str) -> tuple[list[str], list[str]]:
    """Load test data from .src/.tgt or .jsonl files."""
    src_path = Path(test_src)
    ref_path = Path(test_ref)

    # Try JSONL first
    if src_path.suffix == ".jsonl" and src_path.exists():
        sources, references = [], []
        with open(src_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed JSON at line %d", line_num)
                    continue
                if "source" not in rec or "target" not in rec:
                    logger.warning("Missing source/target at line %d", line_num)
                    continue
                sources.append(rec["source"])
                references.append(rec["target"])
        return sources, references

    # Plain text files
    with open(src_path, "r", encoding="utf-8") as f:
        sources = [line.strip() for line in f]
    with open(ref_path, "r", encoding="utf-8") as f:
        references = [line.strip() for line in f]
    return sources, references


def load_model(model_path: str, morphaware: bool, backbone: str, max_length: int):
    """Load a trained model from checkpoint."""
    import torch

    if morphaware:
        from src.model.morphology_aware import MorphologyAwareGEC
        from src.morphology.analyzer import MorphologicalAnalyzer
        analyzer = MorphologicalAnalyzer(use_klpt=False)
        feature_vocab = analyzer.build_feature_vocabulary()
        model = MorphologyAwareGEC(
            model_name=backbone,
            feature_vocab_size=len(feature_vocab),
            max_length=max_length,
        )
    else:
        from src.model.baseline import BaselineGEC
        model = BaselineGEC(model_name=backbone)

    checkpoint = Path(model_path)
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        logger.info("Loaded checkpoint: %s", checkpoint)
    else:
        logger.warning("No checkpoint at %s — using pretrained weights", checkpoint)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    return model


def generate_hypotheses(model, sources: list[str], batch_size: int = 16,
                        num_beams: int = 4) -> list[str]:
    """Generate corrections for all source sentences."""
    import torch
    hypotheses = []
    # Use correct_batch if available for efficiency; fall back to single
    has_batch = hasattr(model, "correct_batch") and callable(model.correct_batch)
    for i in range(0, len(sources), batch_size):
        batch = sources[i:i + batch_size]
        with torch.no_grad():
            if has_batch:
                hyps = model.correct_batch(batch, num_beams=num_beams)
                hypotheses.extend(hyps)
            else:
                for src in batch:
                    hyp = model.correct(src, num_beams=num_beams)
                    hypotheses.append(hyp)
        if (i + batch_size) % 100 == 0 or i + batch_size >= len(sources):
            logger.info("Generated %d/%d corrections", min(i + batch_size, len(sources)), len(sources))
    return hypotheses


def main():
    parser = argparse.ArgumentParser(description="Evaluate GEC model")
    parser.add_argument("--model-path", default="results/models/baseline/best_model.pt")
    parser.add_argument("--backbone", default="google/byt5-small")
    parser.add_argument("--morphaware", action="store_true", default=False)
    parser.add_argument("--test-src", default="data/splits/test.jsonl")
    parser.add_argument("--test-ref", default="data/splits/test.tgt")
    parser.add_argument("--output", default="results/metrics")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load test data
    sources, references = load_test_data(args.test_src, args.test_ref)
    logger.info("Loaded %d test sentences", len(sources))

    # Load model and generate corrections
    logger.info("Loading model from %s (morphaware=%s)", args.model_path, args.morphaware)
    model = load_model(args.model_path, args.morphaware, args.backbone, args.max_length)

    logger.info("Generating corrections...")
    hypotheses = generate_hypotheses(model, sources, args.batch_size, args.num_beams)
    
    # Compute F₀.₅
    logger.info("Computing F₀.₅...")
    metrics = evaluate_corpus(sources, hypotheses, references)
    logger.info("Results: %s", metrics)
    
    # Compute agreement accuracy
    logger.info("Computing agreement accuracy...")
    agreement = evaluate_agreement_accuracy(hypotheses)
    logger.info("Agreement accuracy: %.4f", agreement['accuracy'])
    
    # Save results
    results = {
        "f05": {
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f05": metrics.f05,
            "tp": metrics.tp,
            "fp": metrics.fp,
            "fn": metrics.fn,
        },
        "agreement": agreement,
        "model_path": args.model_path,
        "test_size": len(sources),
    }
    
    results_file = output_dir / "evaluation_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    logger.info("Results saved to %s", results_file)


if __name__ == "__main__":
    main()
