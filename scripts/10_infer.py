"""
Step 10: Single-sentence and batch CLI inference

Runs a trained GEC model from the command line without launching the
Gradio web interface. Supports both baseline and morphology-aware models.

Usage:
    # Single sentence
    python scripts/10_infer.py --model results/models/morphaware/best_model.pt \\
        --morphaware --text "ئەو کتێبەکان خوێندمەوە"

    # Batch (one sentence per line)
    python scripts/10_infer.py --model results/models/baseline/best_model.pt \\
        --input data/test_sentences.txt --output results/corrections.txt

    # Read from stdin
    echo "من دەچین بۆ قوتابخانە" | python scripts/10_infer.py --model results/models/baseline/best_model.pt
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


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
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
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


def main():
    parser = argparse.ArgumentParser(
        description="Sorani Kurdish GEC — command-line inference",
    )
    parser.add_argument("--model", required=True,
                        help="Path to model checkpoint (.pt)")
    parser.add_argument("--morphaware", action="store_true", default=False,
                        help="Use morphology-aware model architecture")
    parser.add_argument("--backbone", default="google/byt5-small",
                        help="Pretrained backbone name")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--beam-width", type=int, default=4,
                        help="Beam search width")
    parser.add_argument("--text", type=str, default=None,
                        help="Single sentence to correct")
    parser.add_argument("--input", type=str, default=None,
                        help="Input file (one sentence per line, UTF-8)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file (default: stdout)")
    parser.add_argument("--json", action="store_true", default=False,
                        help="Output JSON with source and corrected fields")
    parser.add_argument("--ensemble", action="store_true", default=False,
                        help="Ensemble baseline + morphaware via majority vote")
    parser.add_argument("--ensemble-strategy", default="majority_vote",
                        choices=["majority_vote", "best_score"],
                        help="Ensemble combination strategy")
    parser.add_argument("--baseline-path", default="results/models/baseline/best_model.pt",
                        help="Path to baseline checkpoint (used with --ensemble)")
    parser.add_argument("--morphaware-path", default="results/models/morphaware/best_model.pt",
                        help="Path to morphaware checkpoint (used with --ensemble)")
    args = parser.parse_args()

    import torch

    if args.ensemble:
        logger.info("Loading ensemble: baseline=%s, morphaware=%s, strategy=%s",
                     args.baseline_path, args.morphaware_path, args.ensemble_strategy)
        baseline, _, _ = load_model(args.baseline_path, False, args.backbone, args.max_length)
        morphaware, analyzer, feature_extractor = load_model(
            args.morphaware_path, True, args.backbone, args.max_length,
        )
        from src.model.ensemble import EnsembleGEC
        model = EnsembleGEC([baseline, morphaware], strategy=args.ensemble_strategy)
    else:
        model, analyzer, feature_extractor = load_model(
            args.model, args.morphaware, args.backbone, args.max_length,
        )

    # Collect input sentences
    sentences: list[str] = []
    if args.text:
        sentences = [args.text]
    elif args.input:
        with open(args.input, "r", encoding="utf-8") as f:
            sentences = [line.strip() for line in f if line.strip()]
    elif not sys.stdin.isatty():
        sentences = [line.strip() for line in sys.stdin if line.strip()]
    else:
        parser.error("Provide --text, --input, or pipe text via stdin")

    logger.info("Correcting %d sentence(s)...", len(sentences))

    # Generate corrections
    corrections: list[str] = []
    with torch.no_grad():
        if args.ensemble:
            for s in sentences:
                corrections.append(
                    model.correct(s, num_beams=args.beam_width,
                                  analyzer=analyzer,
                                  feature_extractor=feature_extractor)
                )
        elif args.morphaware and analyzer and feature_extractor:
            for s in sentences:
                corrections.append(
                    model.correct_with_morphology(
                        s, analyzer, feature_extractor,
                        num_beams=args.beam_width,
                    )
                )
        elif hasattr(model, "correct_batch") and len(sentences) > 1:
            corrections = model.correct_batch(sentences, num_beams=args.beam_width)
        else:
            for s in sentences:
                corrections.append(model.correct(s, num_beams=args.beam_width))

    # Output
    out_f = open(args.output, "w", encoding="utf-8") if args.output else sys.stdout
    try:
        for src, cor in zip(sentences, corrections):
            if args.json:
                out_f.write(json.dumps(
                    {"source": src, "corrected": cor},
                    ensure_ascii=False,
                ) + "\n")
            else:
                out_f.write(cor + "\n")
    finally:
        if args.output:
            out_f.close()

    if args.output:
        logger.info("Wrote %d corrections to %s", len(corrections), args.output)


if __name__ == "__main__":
    main()
