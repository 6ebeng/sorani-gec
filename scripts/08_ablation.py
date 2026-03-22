"""
Step 8: Ablation Studies

Runs three ablation experiments defined in configs/default.yaml:
  1. no_morphology     — Baseline ByT5 without morphological features
  2. individual_features — One morphological feature at a time
  3. data_size_variation — Training on 10K, 20K, 30K, 40K, 50K pairs

Each experiment trains a model variant, evaluates on the test set, and
saves per-experiment metrics to results/ablation/.

Usage:
    python scripts/08_ablation.py --config configs/default.yaml
    python scripts/08_ablation.py --experiment no_morphology
    python scripts/08_ablation.py --experiment data_size_variation --data-sizes 10000 30000 50000
"""

import argparse
import copy
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


MORPH_FEATURES = [
    "person", "number", "tense", "aspect", "case",
    "definiteness", "transitivity", "clitic_person", "clitic_number",
]

DEFAULT_DATA_SIZES = [10_000, 20_000, 30_000, 40_000, 50_000]


def load_config(config_path: str) -> dict:
    """Load YAML config."""
    import yaml  # deferred: not needed at import time
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_test_data(test_path: str) -> tuple[list[str], list[str]]:
    """Load test pairs from JSONL or plain text files."""
    path = Path(test_path)
    sources, references = [], []
    if path.suffix == ".jsonl" and path.exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                sources.append(rec["source"])
                references.append(rec["target"])
    else:
        src_path = path.parent / (path.stem + ".src")
        ref_path = path.parent / (path.stem + ".tgt")
        with open(src_path, "r", encoding="utf-8") as f:
            sources = [l.strip() for l in f]
        with open(ref_path, "r", encoding="utf-8") as f:
            references = [l.strip() for l in f]
    return sources, references


def evaluate_model(model, sources: list[str], references: list[str],
                   batch_size: int = 16) -> dict:
    """Run F0.5 and agreement accuracy on model outputs."""
    import torch
    model.eval()
    hypotheses = []
    with torch.no_grad():
        for i in range(0, len(sources), batch_size):
            batch = sources[i:i + batch_size]
            if hasattr(model, "correct_batch") and callable(model.correct_batch):
                hypotheses.extend(model.correct_batch(batch))
            else:
                for src in batch:
                    hypotheses.append(model.correct(src))

    metrics = evaluate_corpus(sources, hypotheses, references)
    agreement = evaluate_agreement_accuracy(hypotheses)

    return {
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f05": metrics.f05,
        "tp": metrics.tp,
        "fp": metrics.fp,
        "fn": metrics.fn,
        "agreement_accuracy": agreement["accuracy"],
        "agreement_correct": agreement["correct_sentences"],
        "agreement_total": agreement["total_sentences"],
    }


def train_model(config: dict, use_morphology: bool,
                feature_subset: list[str] | None = None,
                data_size: int | None = None,
                output_dir: Path | None = None) -> object:
    """Train a model variant and return the trained model.

    Parameters:
        config: Full config dict.
        use_morphology: Whether to include morphological features.
        feature_subset: If set, only these features are enabled.
        data_size: If set, subsample training data to this count.
        output_dir: Where to save checkpoints.
    """
    import torch
    from src.data.splitter import load_pairs

    cfg = copy.deepcopy(config)
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Load training data
    train_path = Path(cfg["data"]["splits_dir"]) / "train.jsonl"
    pairs = load_pairs(train_path)
    if data_size and data_size < len(pairs):
        import random
        rng = random.Random(cfg.get("data", {}).get("seed", 42))
        pairs = rng.sample(pairs, data_size)
    sources = [p["source"] for p in pairs]
    targets = [p["target"] for p in pairs]

    backbone = cfg["model"].get("pretrained", "google/byt5-small")
    max_length = cfg["data"].get("max_seq_length", 128)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if use_morphology:
        from src.model.morphology_aware import MorphologyAwareGEC
        from src.morphology.analyzer import MorphologicalAnalyzer

        analyzer = MorphologicalAnalyzer(use_klpt=False)
        feature_vocab = analyzer.build_feature_vocabulary()

        model = MorphologyAwareGEC(
            model_name=backbone,
            feature_vocab_size=len(feature_vocab),
            max_length=max_length,
        )

        # If feature_subset is provided, zero out features not in subset
        if feature_subset:
            active = set(feature_subset)
            logger.info("Active features: %s", active)
            # Store feature mask on model for training loop to use
            model._ablation_active_features = active
    else:
        from src.model.baseline import BaselineGEC
        model = BaselineGEC(model_name=backbone)

    model = model.to(device)

    # Training loop — follows scripts/06_train_morphaware.py conventions
    epochs = cfg["training"].get("max_epochs", 30)
    batch_size = cfg["training"].get("batch_size", 32)
    lr = cfg["training"].get("learning_rate", 5e-5)
    patience = cfg["training"].get("early_stopping_patience", 5)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=cfg["training"].get("weight_decay", 0.01))

    best_loss = float("inf")
    stale = 0

    logger.info("Training on %d pairs for up to %d epochs (batch_size=%d)",
                len(sources), epochs, batch_size)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, len(sources), batch_size):
            batch_src = sources[i:i + batch_size]
            batch_tgt = targets[i:i + batch_size]

            loss = model.training_step(batch_src, batch_tgt)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        logger.info("Epoch %d/%d — avg loss: %.4f", epoch, epochs, avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            stale = 0
            if output_dir:
                torch.save(model.state_dict(), output_dir / "best_model.pt")
        else:
            stale += 1
            if stale >= patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

    return model


# ============================================================================
# Ablation Experiments
# ============================================================================

def run_no_morphology(config: dict, sources: list[str],
                      references: list[str], output_dir: Path) -> dict:
    """Experiment 1: Baseline without morphological features."""
    logger.info("=== Ablation: no_morphology ===")
    exp_dir = output_dir / "no_morphology"

    model = train_model(config, use_morphology=False, output_dir=exp_dir)
    results = evaluate_model(model, sources, references)
    results["experiment"] = "no_morphology"

    with open(exp_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("no_morphology — F0.5=%.4f, agreement=%.4f",
                results["f05"], results["agreement_accuracy"])
    return results


def run_individual_features(config: dict, sources: list[str],
                            references: list[str],
                            output_dir: Path) -> list[dict]:
    """Experiment 2: One morphological feature at a time."""
    logger.info("=== Ablation: individual_features ===")
    all_results = []

    for feature in MORPH_FEATURES:
        logger.info("--- Feature: %s ---", feature)
        exp_dir = output_dir / "individual_features" / feature
        model = train_model(config, use_morphology=True,
                            feature_subset=[feature], output_dir=exp_dir)
        results = evaluate_model(model, sources, references)
        results["experiment"] = "individual_features"
        results["feature"] = feature

        with open(exp_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info("  %s — F0.5=%.4f, agreement=%.4f",
                     feature, results["f05"], results["agreement_accuracy"])
        all_results.append(results)

    return all_results


def run_data_size_variation(config: dict, sources: list[str],
                            references: list[str], output_dir: Path,
                            sizes: list[int] | None = None) -> list[dict]:
    """Experiment 3: Varying training data size."""
    logger.info("=== Ablation: data_size_variation ===")
    if sizes is None:
        sizes = DEFAULT_DATA_SIZES
    all_results = []

    for size in sizes:
        logger.info("--- Data size: %d ---", size)
        exp_dir = output_dir / "data_size_variation" / f"{size}"
        model = train_model(config, use_morphology=True,
                            data_size=size, output_dir=exp_dir)
        results = evaluate_model(model, sources, references)
        results["experiment"] = "data_size_variation"
        results["data_size"] = size

        with open(exp_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info("  %d pairs — F0.5=%.4f, agreement=%.4f",
                     size, results["f05"], results["agreement_accuracy"])
        all_results.append(results)

    return all_results


def main():
    parser = argparse.ArgumentParser(description="GEC Ablation Studies")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--experiment", default="all",
                        choices=["all", "no_morphology", "individual_features",
                                 "data_size_variation"])
    parser.add_argument("--test-data", default="data/splits/test.jsonl")
    parser.add_argument("--output", default="results/ablation")
    parser.add_argument("--data-sizes", nargs="*", type=int, default=None,
                        help="Custom data sizes for data_size_variation")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    sources, references = load_test_data(args.test_data)
    logger.info("Loaded %d test sentences", len(sources))

    summary = {}

    if args.experiment in ("all", "no_morphology"):
        r = run_no_morphology(config, sources, references, output_dir)
        summary["no_morphology"] = r

    if args.experiment in ("all", "individual_features"):
        r = run_individual_features(config, sources, references, output_dir)
        summary["individual_features"] = r

    if args.experiment in ("all", "data_size_variation"):
        r = run_data_size_variation(config, sources, references, output_dir,
                                    sizes=args.data_sizes)
        summary["data_size_variation"] = r

    # Save combined summary
    summary_file = output_dir / "ablation_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Ablation summary saved to %s", summary_file)


if __name__ == "__main__":
    main()
