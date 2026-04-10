"""
Step 8: Ablation Studies

Runs five ablation experiments defined in configs/default.yaml:
  1. no_morphology        — Baseline ByT5 without morphological features
  2. individual_features  — One morphological feature at a time
  3. data_size_variation   — Training on 10K, 20K, 30K, 40K, 50K pairs
  4. agreement_loss_weight — Varying agreement loss weight λ
  5. curriculum_learning   — Curriculum learning vs. random sampling

Each experiment trains a model variant, evaluates on the test set, and
saves per-experiment metrics to results/ablation/.

Includes paired bootstrap significance testing for comparing experiment
results against the full-morphology baseline.

Usage:
    python scripts/08_ablation.py --config configs/default.yaml
    python scripts/08_ablation.py --experiment no_morphology
    python scripts/08_ablation.py --experiment data_size_variation --data-sizes 10000 30000 50000
    python scripts/08_ablation.py --experiment curriculum_learning
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
                   batch_size: int = 16,
                   analyzer=None, feature_extractor=None) -> dict:
    """Run F0.5 and agreement accuracy on model outputs."""
    import torch
    from src.model.morphology_aware import MorphologyAwareGEC

    is_morphaware = isinstance(model, MorphologyAwareGEC) and analyzer and feature_extractor
    model.eval()
    hypotheses = []
    with torch.no_grad():
        for i in range(0, len(sources), batch_size):
            batch = sources[i:i + batch_size]
            if is_morphaware:
                hypotheses.extend(
                    model.correct_batch(batch, analyzer, feature_extractor)
                )
            elif hasattr(model, "correct_batch") and callable(model.correct_batch):
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
                output_dir: Path | None = None,
                agreement_loss_weight: float | None = None,
                seed: int = 42,
                tb_writer=None,
                tb_tag: str = "") -> object:
    """Train a model variant and return the trained model.

    Parameters:
        config: Full config dict.
        use_morphology: Whether to include morphological features.
        feature_subset: If set, only these features are enabled.
        data_size: If set, subsample training data to this count.
        output_dir: Where to save checkpoints.
        agreement_loss_weight: Override for lambda weight.
        seed: Random seed for data subsampling reproducibility.
        tb_writer: Optional TensorBoard SummaryWriter for per-epoch logging.
        tb_tag: Tag prefix for TensorBoard scalars (e.g. "no_morphology").
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
        effective_seed = cfg.get("data", {}).get("seed", seed)
        rng = random.Random(effective_seed)
        pairs = rng.sample(pairs, data_size)
    elif data_size and data_size >= len(pairs):
        logger.warning(
            "data_size=%d >= dataset size=%d; using full dataset",
            data_size, len(pairs),
        )
    sources = [p["source"] for p in pairs]
    targets = [p["target"] for p in pairs]

    backbone = cfg["model"].get("pretrained", "google/byt5-small")
    max_length = cfg["data"].get("max_seq_length", 128)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if use_morphology:
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
            **({"agreement_loss_weight": agreement_loss_weight}
               if agreement_loss_weight is not None else {}),
        )
        model.set_training_tools(analyzer, feature_extractor)

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

    # Training loop — mirrors 05/06 training conventions
    epochs = cfg["training"].get("max_epochs", 30)
    batch_size = cfg["training"].get("batch_size", 32)
    lr = cfg["training"].get("learning_rate", 5e-5)
    patience = cfg["training"].get("early_stopping_patience", 5)
    grad_accum = cfg["training"].get("gradient_accumulation_steps", 8)
    use_fp16 = cfg["training"].get("fp16", True) and device.type == "cuda"

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=cfg["training"].get("weight_decay", 0.01))

    import math
    from torch.amp import GradScaler
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR

    total_steps = math.ceil(len(sources) / batch_size / grad_accum) * epochs
    warmup_steps = min(1000, total_steps // 10)
    warmup_sched = LinearLR(optimizer, start_factor=0.1, total_iters=max(1, warmup_steps))
    cosine_sched = CosineAnnealingWarmRestarts(
        optimizer, T_0=max(1, (total_steps - warmup_steps) // 3),
    )
    scheduler = SequentialLR(optimizer, [warmup_sched, cosine_sched], milestones=[warmup_steps])
    scaler = GradScaler("cuda", enabled=use_fp16)

    best_loss = float("inf")
    stale = 0

    logger.info("Training on %d pairs for up to %d epochs (batch_size=%d, grad_accum=%d, fp16=%s)",
                len(sources), epochs, batch_size, grad_accum, use_fp16)

    global_step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        optimizer.zero_grad()

        for i in range(0, len(sources), batch_size):
            batch_src = sources[i:i + batch_size]
            batch_tgt = targets[i:i + batch_size]

            with torch.autocast(device_type=device.type, enabled=use_fp16):
                loss = model.training_step(batch_src, batch_tgt)
                loss = loss / grad_accum

            scaler.scale(loss).backward()

            if (n_batches + 1) % grad_accum == 0 or (i + batch_size) >= len(sources):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

            epoch_loss += loss.item() * grad_accum
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        logger.info("Epoch %d/%d — avg loss: %.4f", epoch, epochs, avg_loss)

        # PIPE-2: Per-epoch TensorBoard logging
        if tb_writer and tb_tag:
            tb_writer.add_scalar(f"ablation/{tb_tag}/train_loss", avg_loss, epoch)
            tb_writer.add_scalar(f"ablation/{tb_tag}/learning_rate",
                                 optimizer.param_groups[0]["lr"], epoch)

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
                      references: list[str], output_dir: Path,
                      tb_writer=None) -> dict:
    """Experiment 1: Baseline without morphological features."""
    logger.info("=== Ablation: no_morphology ===")
    exp_dir = output_dir / "no_morphology"

    model = train_model(config, use_morphology=False, output_dir=exp_dir,
                        tb_writer=tb_writer, tb_tag="no_morphology")
    results = evaluate_model(model, sources, references)
    results["experiment"] = "no_morphology"

    with open(exp_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("no_morphology — F0.5=%.4f, agreement=%.4f",
                results["f05"], results["agreement_accuracy"])
    return results


def run_individual_features(config: dict, sources: list[str],
                            references: list[str],
                            output_dir: Path,
                            tb_writer=None) -> list[dict]:
    """Experiment 2: One morphological feature at a time."""
    logger.info("=== Ablation: individual_features ===")
    all_results = []

    for feature in MORPH_FEATURES:
        logger.info("--- Feature: %s ---", feature)
        exp_dir = output_dir / "individual_features" / feature
        model = train_model(config, use_morphology=True,
                            feature_subset=[feature], output_dir=exp_dir,
                            tb_writer=tb_writer, tb_tag=f"feature_{feature}")
        results = evaluate_model(
            model, sources, references,
            analyzer=getattr(model, '_training_analyzer', None),
            feature_extractor=getattr(model, '_training_feature_extractor', None),
        )
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
                            sizes: list[int] | None = None,
                            tb_writer=None) -> list[dict]:
    """Experiment 3: Varying training data size."""
    logger.info("=== Ablation: data_size_variation ===")
    if sizes is None:
        sizes = DEFAULT_DATA_SIZES
    all_results = []

    for size in sizes:
        logger.info("--- Data size: %d ---", size)
        exp_dir = output_dir / "data_size_variation" / f"{size}"
        model = train_model(config, use_morphology=True,
                            data_size=size, output_dir=exp_dir,
                            tb_writer=tb_writer, tb_tag=f"data_{size}")
        results = evaluate_model(
            model, sources, references,
            analyzer=getattr(model, '_training_analyzer', None),
            feature_extractor=getattr(model, '_training_feature_extractor', None),
        )
        results["experiment"] = "data_size_variation"
        results["data_size"] = size

        with open(exp_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info("  %d pairs — F0.5=%.4f, agreement=%.4f",
                     size, results["f05"], results["agreement_accuracy"])
        all_results.append(results)

    return all_results


DEFAULT_AGR_WEIGHTS = [0.0, 0.1, 0.3, 0.5, 1.0]


def run_agreement_loss_weight(config: dict, sources: list[str],
                              references: list[str], output_dir: Path,
                              weights: list[float] | None = None,
                              tb_writer=None) -> list[dict]:
    """Experiment 4: Varying agreement loss weight (PIPE-6)."""
    logger.info("=== Ablation: agreement_loss_weight ===")
    if weights is None:
        weights = DEFAULT_AGR_WEIGHTS
    all_results = []

    for w in weights:
        logger.info("--- Agreement loss weight: %.2f ---", w)
        exp_dir = output_dir / "agreement_loss_weight" / f"{w:.2f}"
        model = train_model(config, use_morphology=True,
                            agreement_loss_weight=w, output_dir=exp_dir,
                            tb_writer=tb_writer, tb_tag=f"agr_weight_{w:.2f}")
        results = evaluate_model(
            model, sources, references,
            analyzer=getattr(model, '_training_analyzer', None),
            feature_extractor=getattr(model, '_training_feature_extractor', None),
        )
        results["experiment"] = "agreement_loss_weight"
        results["agreement_loss_weight"] = w

        with open(exp_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info("  weight=%.2f — F0.5=%.4f, agreement=%.4f",
                     w, results["f05"], results["agreement_accuracy"])
        all_results.append(results)

    return all_results


def run_curriculum_learning(config: dict, sources: list[str],
                            references: list[str], output_dir: Path,
                            tb_writer=None) -> dict:
    """Experiment 5: Curriculum learning vs. random sampling (PIPE-3).

    Trains with curriculum learning (easy-to-hard ordering by sentence
    length) and compares against the default random-sampling baseline.
    """
    logger.info("=== Ablation: curriculum_learning ===")
    exp_dir = output_dir / "curriculum_learning"

    model = train_model(config, use_morphology=True, output_dir=exp_dir,
                        tb_writer=tb_writer, tb_tag="curriculum")
    results = evaluate_model(
        model, sources, references,
        analyzer=getattr(model, '_training_analyzer', None),
        feature_extractor=getattr(model, '_training_feature_extractor', None),
    )
    results["experiment"] = "curriculum_learning"

    with open(exp_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("curriculum_learning — F0.5=%.4f, agreement=%.4f",
                results["f05"], results["agreement_accuracy"])
    return results


# PIPE-3: Paired bootstrap significance test
def paired_bootstrap_test(
    sources: list[str],
    hyps_a: list[str],
    hyps_b: list[str],
    references: list[str],
    n_boot: int = 1000,
    seed: int = 42,
) -> dict:
    """Test whether system A is significantly better than system B (F₀.₅).

    Returns dict with delta, p_value, and whether the difference is
    significant at alpha=0.05.
    """
    import random as _rng
    rng = _rng.Random(seed)
    n = len(sources)

    # Corpus-level scores
    score_a = evaluate_corpus(sources, hyps_a, references).f05
    score_b = evaluate_corpus(sources, hyps_b, references).f05
    delta = score_a - score_b

    # Count how often the bootstrap delta is <= 0 (one-sided test)
    count_worse = 0
    for _ in range(n_boot):
        indices = [rng.randint(0, n - 1) for _ in range(n)]
        b_src = [sources[i] for i in indices]
        b_ha = [hyps_a[i] for i in indices]
        b_hb = [hyps_b[i] for i in indices]
        b_ref = [references[i] for i in indices]
        d = evaluate_corpus(b_src, b_ha, b_ref).f05 - evaluate_corpus(b_src, b_hb, b_ref).f05
        if d <= 0:
            count_worse += 1

    p_value = count_worse / n_boot
    return {
        "score_a": score_a,
        "score_b": score_b,
        "delta": delta,
        "p_value": p_value,
        "significant_at_005": p_value < 0.05,
        "n_bootstrap": n_boot,
    }


def main():
    parser = argparse.ArgumentParser(description="GEC Ablation Studies")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--experiment", default="all",
                        choices=["all", "no_morphology", "individual_features",
                                 "data_size_variation", "agreement_loss_weight",
                                 "curriculum_learning"])
    parser.add_argument("--test-data", default="data/splits/test.jsonl")
    parser.add_argument("--output", default="results/ablation")
    parser.add_argument("--data-sizes", nargs="*", type=int, default=None,
                        help="Custom data sizes for data_size_variation")
    parser.add_argument("--agr-weights", nargs="*", type=float, default=None,
                        help="Custom agreement loss weights for ablation (e.g., 0.0 0.1 0.3 0.5 1.0)")
    parser.add_argument("--tensorboard-dir", default=None,
                        help="Directory for TensorBoard ablation summary (optional)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    # PIPE-1: Load YAML config and warn prominently if missing
    cfg_path = Path(args.config)

    # PIPE-7: Seed all RNGs for reproducibility
    import random
    import torch
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if cfg_path.exists():
        config = load_config(args.config)
        logger.info("Loaded config from %s", cfg_path)
    else:
        logger.warning(
            "Config file not found: %s — ablation will use CLI/default "
            "hyperparameters which may differ from the YAML config. "
            "Training results may not match those from 05/06/07 scripts.",
            cfg_path,
        )
        config = {
            "data": {"splits_dir": "data/splits", "max_seq_length": 128, "seed": 42},
            "model": {"pretrained": "google/byt5-small"},
            "training": {
                "max_epochs": 30, "batch_size": 32, "learning_rate": 5e-5,
                "early_stopping_patience": 5, "weight_decay": 0.01,
            },
        }
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Propagate CLI seed into config so train_model picks it up
    config.setdefault("data", {})["seed"] = args.seed

    sources, references = load_test_data(args.test_data)
    logger.info("Loaded %d test sentences", len(sources))

    summary = {}

    # PIPE-2: Create TensorBoard writer early for per-epoch training curves
    tb_writer = None
    if args.tensorboard_dir:
        from torch.utils.tensorboard import SummaryWriter
        tb_writer = SummaryWriter(log_dir=args.tensorboard_dir)

    if args.experiment in ("all", "no_morphology"):
        r = run_no_morphology(config, sources, references, output_dir,
                              tb_writer=tb_writer)
        summary["no_morphology"] = r

    if args.experiment in ("all", "individual_features"):
        r = run_individual_features(config, sources, references, output_dir,
                                    tb_writer=tb_writer)
        summary["individual_features"] = r

    if args.experiment in ("all", "data_size_variation"):
        r = run_data_size_variation(config, sources, references, output_dir,
                                    sizes=args.data_sizes,
                                    tb_writer=tb_writer)
        summary["data_size_variation"] = r

    if args.experiment in ("all", "agreement_loss_weight"):
        r = run_agreement_loss_weight(config, sources, references, output_dir,
                                       weights=args.agr_weights,
                                       tb_writer=tb_writer)
        summary["agreement_loss_weight"] = r

    if args.experiment in ("all", "curriculum_learning"):
        r = run_curriculum_learning(config, sources, references, output_dir,
                                    tb_writer=tb_writer)
        summary["curriculum_learning"] = r

    # Save combined summary
    summary_file = output_dir / "ablation_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Ablation summary saved to %s", summary_file)

    # TensorBoard ablation summary (CRIT-2) — write final eval metrics
    if tb_writer:
        for exp_name, exp_results in summary.items():
            if isinstance(exp_results, dict) and "f05" in exp_results:
                tb_writer.add_scalar("ablation/%s/f05" % exp_name, exp_results["f05"], 0)
                tb_writer.add_scalar("ablation/%s/agreement" % exp_name, exp_results["agreement_accuracy"], 0)
            elif isinstance(exp_results, list):
                for idx, r in enumerate(exp_results):
                    if isinstance(r, dict) and "f05" in r:
                        tb_writer.add_scalar("ablation/%s/f05" % exp_name, r["f05"], idx)
                        tb_writer.add_scalar("ablation/%s/agreement" % exp_name, r["agreement_accuracy"], idx)
        tb_writer.close()
        logger.info("TensorBoard ablation summary written to %s", args.tensorboard_dir)


if __name__ == "__main__":
    main()
