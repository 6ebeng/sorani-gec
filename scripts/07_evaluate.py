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

from src.evaluation.f05_scorer import (
    evaluate_corpus,
    evaluate_corpus_by_type,
    evaluate_corpus_span,
    evaluate_corpus_with_sentences,
)
from src.evaluation.agreement_accuracy import evaluate_agreement_accuracy
from src.evaluation.gleu_scorer import compute_gleu

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_test_data(test_src: str, test_ref: str) -> tuple[list[str], list[str], list[dict]]:
    """Load test data from .src/.tgt or .jsonl files."""
    src_path = Path(test_src)
    ref_path = Path(test_ref)

    # Try JSONL first
    if src_path.suffix == ".jsonl" and src_path.exists():
        sources, references, records = [], [], []
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
                records.append(rec)
        return sources, references, records

    # Plain text files
    with open(src_path, "r", encoding="utf-8") as f:
        sources = [line.strip() for line in f]
    with open(ref_path, "r", encoding="utf-8") as f:
        references = [line.strip() for line in f]
    return sources, references, []


def load_model(model_path: str, morphaware: bool, backbone: str, max_length: int):
    """Load a trained model from checkpoint.

    Returns:
        (model, analyzer, feature_extractor) — analyzer and feature_extractor
        are None for baseline models.
    """
    import torch

    analyzer = None
    feature_extractor = None

    if morphaware:
        from src.model.morphology_aware import MorphologyAwareGEC
        from src.morphology.analyzer import MorphologicalAnalyzer
        from src.morphology.features import FeatureExtractor
        analyzer = MorphologicalAnalyzer(use_klpt=False)
        feature_vocab = analyzer.build_feature_vocabulary()
        feature_extractor = FeatureExtractor(analyzer=analyzer)
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
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)
        logger.info("Loaded checkpoint: %s", checkpoint)
    else:
        logger.warning("No checkpoint at %s — using pretrained weights", checkpoint)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    return model, analyzer, feature_extractor


def generate_hypotheses(model, sources: list[str], batch_size: int = 16,
                        num_beams: int = 4,
                        analyzer=None, feature_extractor=None) -> list[str]:
    """Generate corrections for all source sentences."""
    import torch
    from src.model.morphology_aware import MorphologyAwareGEC

    is_morphaware = isinstance(model, MorphologyAwareGEC) and analyzer and feature_extractor
    hypotheses = []
    for i in range(0, len(sources), batch_size):
        batch = sources[i:i + batch_size]
        with torch.no_grad():
            if is_morphaware:
                hyps = model.correct_batch(
                    batch, analyzer, feature_extractor, num_beams=num_beams,
                )
                hypotheses.extend(hyps)
            elif hasattr(model, "correct_batch") and callable(model.correct_batch):
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
    parser.add_argument("--config", default="configs/default.yaml",
                        help="Path to YAML config (reads max_seq_length, beam, etc.)")
    parser.add_argument("--model-path", default="results/models/baseline/best_model.pt")
    parser.add_argument("--backbone", default="google/byt5-small")
    parser.add_argument("--morphaware", action="store_true", default=False)
    parser.add_argument("--test-src", default="data/splits/test.jsonl")
    parser.add_argument("--test-ref", default="data/splits/test.tgt")
    parser.add_argument("--output", default="results/metrics")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--spell-check", action="store_true", default=False,
                        help="Apply spell-check post-processing to model output")
    args = parser.parse_args()

    # PIPE-8: Load YAML config and apply as defaults when CLI was not
    # explicitly overridden — matches the pattern in 05/06 scripts.
    cfg_path = Path(args.config)
    if cfg_path.exists():
        import yaml
        with open(cfg_path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
        _yaml_defaults = {
            "max_length": cfg.get("data", {}).get("max_seq_length"),
            "batch_size": cfg.get("training", {}).get("batch_size"),
        }
        sentinel = parser.parse_args([])
        for key, yaml_val in _yaml_defaults.items():
            if yaml_val is not None and getattr(args, key) == getattr(sentinel, key):
                setattr(args, key, yaml_val)
        logger.info("Loaded config from %s", cfg_path)
    else:
        logger.warning("Config file not found: %s — using CLI defaults", cfg_path)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load test data
    sources, references, records = load_test_data(args.test_src, args.test_ref)
    logger.info("Loaded %d test sentences", len(sources))

    # Load model and generate corrections
    logger.info("Loading model from %s (morphaware=%s)", args.model_path, args.morphaware)
    model, analyzer, feature_extractor = load_model(
        args.model_path, args.morphaware, args.backbone, args.max_length,
    )

    logger.info("Generating corrections...")
    hypotheses = generate_hypotheses(
        model, sources, args.batch_size, args.num_beams,
        analyzer=analyzer, feature_extractor=feature_extractor,
    )

    # Optional spell-check post-processing
    if args.spell_check:
        from src.data.spell_checker import SoraniSpellChecker
        checker = SoraniSpellChecker()
        if checker.is_available():
            logger.info("Applying spell-check post-processing...")
            hypotheses = [checker.correct_sentence(h) for h in hypotheses]
        else:
            logger.warning("Spell checker not available — skipping post-processing")
    
    # Compute F₀.₅ (corpus + sentence-level)
    logger.info("Computing F₀.₅...")
    metrics, sentence_metrics = evaluate_corpus_with_sentences(
        sources, hypotheses, references,
    )
    logger.info("Results: %s", metrics)

    # Identify hardest sentences (lowest sentence-level F₀.₅)
    hardest_indices = sorted(
        range(len(sentence_metrics)),
        key=lambda k: sentence_metrics[k].f05,
    )[:10]
    logger.info("Hardest sentences (lowest F₀.₅):")
    for idx in hardest_indices:
        sm = sentence_metrics[idx]
        logger.info("  [%d] F₀.₅=%.4f  src=%s", idx, sm.f05, sources[idx][:80])

    # Span-based evaluation (morphological vs substitution breakdown)
    logger.info("Computing span-based metrics...")
    span_metrics, span_per_type = evaluate_corpus_span(
        sources, hypotheses, references,
    )
    logger.info("Span-based results: %s", span_metrics)
    for etype, em in span_per_type.items():
        logger.info("  %s: %s", etype, em)
    
    # Per-error-type F₀.₅ breakdown (if error types available)
    per_type_results = {}
    if records and records[0].get("error_type"):
        error_types = [r.get("error_type", "unknown") for r in records]
        per_type = evaluate_corpus_by_type(sources, hypotheses, references, error_types)
        for etype, m in per_type.items():
            logger.info("  %s: %s", etype, m)
            per_type_results[etype] = {
                "precision": m.precision, "recall": m.recall, "f05": m.f05,
                "tp": m.tp, "fp": m.fp, "fn": m.fn,
            }
    else:
        logger.warning("error_type field missing from test data — per-type F0.5 breakdown skipped")
    
    # Compute GLEU
    logger.info("Computing GLEU...")
    gleu_score = compute_gleu(sources, hypotheses, references)
    logger.info("GLEU: %.4f", gleu_score)

    # Compute agreement accuracy
    logger.info("Computing agreement accuracy...")
    agreement = evaluate_agreement_accuracy(hypotheses)
    logger.info("Agreement accuracy: %.4f", agreement['accuracy'])
    
    # Save hypotheses for manual inspection
    hyp_file = output_dir / "hypotheses.txt"
    with open(hyp_file, "w", encoding="utf-8") as f:
        for h in hypotheses:
            f.write(h + "\n")
    logger.info("Hypotheses saved to %s", hyp_file)

    # Save evaluation pairs as JSONL (consumed by web/evaluation.py)
    eval_pairs_file = output_dir / "evaluation_pairs.jsonl"
    with open(eval_pairs_file, "w", encoding="utf-8") as f:
        for src, hyp in zip(sources, hypotheses):
            f.write(json.dumps(
                {"source": src, "corrected": hyp},
                ensure_ascii=False,
            ) + "\n")
    logger.info("Evaluation pairs saved to %s", eval_pairs_file)
    
    # Save results
    # Sentence-level F₀.₅ distribution stats
    sent_f05_scores = [sm.f05 for sm in sentence_metrics]
    sent_stats = {
        "mean": sum(sent_f05_scores) / len(sent_f05_scores) if sent_f05_scores else 0.0,
        "min": min(sent_f05_scores) if sent_f05_scores else 0.0,
        "max": max(sent_f05_scores) if sent_f05_scores else 0.0,
        "zero_count": sum(1 for s in sent_f05_scores if s == 0.0),
        "perfect_count": sum(1 for s in sent_f05_scores if s == 1.0),
    }

    results = {
        "f05": {
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f05": metrics.f05,
            "tp": metrics.tp,
            "fp": metrics.fp,
            "fn": metrics.fn,
        },
        "f05_sentence_stats": sent_stats,
        "f05_per_type": per_type_results,
        "f05_span": {
            "overall": {
                "precision": span_metrics.precision,
                "recall": span_metrics.recall,
                "f05": span_metrics.f05,
                "tp": span_metrics.tp, "fp": span_metrics.fp, "fn": span_metrics.fn,
            },
            "per_edit_type": {
                etype: {
                    "precision": m.precision, "recall": m.recall, "f05": m.f05,
                    "tp": m.tp, "fp": m.fp, "fn": m.fn,
                }
                for etype, m in span_per_type.items()
            },
        },
        "gleu": gleu_score,
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
