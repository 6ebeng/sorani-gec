"""
Step 12: Hyperparameter Search

Runs a grid or random search over key hyperparameters for both baseline
and morphology-aware models.  Results are written to a JSON file for
analysis and thesis reporting.

Search dimensions (configurable via CLI):
  - Learning rate: {1e-5, 3e-5, 5e-5, 1e-4}
  - Batch size: {8, 16, 32}
  - Agreement loss weight (morphaware only): {0.1, 0.3, 0.5}
  - Gradient accumulation steps: {4, 8}

Usage:
    python scripts/12_hyperparam_search.py --config configs/default.yaml
    python scripts/12_hyperparam_search.py --strategy random --budget 10
    python scripts/12_hyperparam_search.py --model-type morphaware --lr 1e-5 3e-5 5e-5
"""

import argparse
import copy
import itertools
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_LR = [1e-5, 3e-5, 5e-5, 1e-4]
DEFAULT_BATCH_SIZES = [8, 16, 32]
DEFAULT_AGR_WEIGHTS = [0.1, 0.3, 0.5]
DEFAULT_GRAD_ACCUM = [4, 8]


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_search_space(args) -> list[dict]:
    """Build the full grid of hyperparameter combinations."""
    lrs = args.lr or DEFAULT_LR
    batch_sizes = args.batch_size_list or DEFAULT_BATCH_SIZES
    grad_accums = args.grad_accum_list or DEFAULT_GRAD_ACCUM

    if args.model_type == "morphaware":
        agr_weights = args.agr_weights or DEFAULT_AGR_WEIGHTS
        combos = list(itertools.product(lrs, batch_sizes, grad_accums, agr_weights))
        space = [
            {
                "lr": lr,
                "batch_size": bs,
                "grad_accum_steps": ga,
                "agreement_loss_weight": aw,
            }
            for lr, bs, ga, aw in combos
        ]
    else:
        combos = list(itertools.product(lrs, batch_sizes, grad_accums))
        space = [
            {
                "lr": lr,
                "batch_size": bs,
                "grad_accum_steps": ga,
            }
            for lr, bs, ga in combos
        ]

    return space


def run_single_trial(
    trial_id: int,
    hparams: dict,
    args,
    base_cfg: dict,
) -> dict:
    """Train and evaluate a single hyperparameter configuration."""
    import torch
    from src.evaluation.f05_scorer import evaluate_corpus

    cfg = copy.deepcopy(base_cfg)
    cfg.setdefault("training", {})
    cfg["training"]["learning_rate"] = hparams["lr"]
    cfg["training"]["batch_size"] = hparams["batch_size"]
    cfg["training"]["gradient_accumulation_steps"] = hparams["grad_accum_steps"]

    effective_batch = hparams["batch_size"] * hparams["grad_accum_steps"]

    logger.info(
        "Trial %d/%d: lr=%.1e, batch=%d, grad_accum=%d (eff_batch=%d)%s",
        trial_id,
        args.total_trials,
        hparams["lr"],
        hparams["batch_size"],
        hparams["grad_accum_steps"],
        effective_batch,
        " agr_w=%.2f" % hparams["agreement_loss_weight"]
        if "agreement_loss_weight" in hparams
        else "",
    )

    trial_dir = Path(args.output) / f"trial_{trial_id:03d}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    # Save trial config
    with open(trial_dir / "hparams.json", "w", encoding="utf-8") as f:
        json.dump(hparams, f, indent=2)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(cfg.get("data", {}).get("splits_dir", "data/splits"))
    max_length = cfg.get("data", {}).get("max_seq_length", 256)

    # Load data
    train_jsonl = data_dir / "train.jsonl"
    dev_jsonl = data_dir / "dev.jsonl"

    if not train_jsonl.exists():
        logger.error("Training data not found: %s", train_jsonl)
        return {"trial_id": trial_id, "hparams": hparams, "error": "no training data"}

    train_sources, train_targets = _load_pairs(train_jsonl)
    dev_sources, dev_targets = _load_pairs(dev_jsonl)

    # Subsample for faster search if requested
    if args.subsample and args.subsample < len(train_sources):
        indices = random.sample(range(len(train_sources)), args.subsample)
        train_sources = [train_sources[i] for i in indices]
        train_targets = [train_targets[i] for i in indices]
        logger.info("Subsampled training data to %d pairs", args.subsample)

    # Reduced epochs for search
    n_epochs = args.search_epochs

    start_time = time.time()

    if args.model_type == "morphaware":
        result = _train_morphaware_trial(
            train_sources, train_targets,
            dev_sources, dev_targets,
            hparams, cfg, device, max_length, n_epochs,
            trial_dir, args,
        )
    else:
        result = _train_baseline_trial(
            train_sources, train_targets,
            dev_sources, dev_targets,
            hparams, cfg, device, max_length, n_epochs,
            trial_dir, args,
        )

    elapsed = time.time() - start_time
    result["trial_id"] = trial_id
    result["hparams"] = hparams
    result["elapsed_seconds"] = round(elapsed, 1)
    result["effective_batch_size"] = effective_batch

    logger.info(
        "Trial %d done: val_f05=%.4f, elapsed=%.0fs",
        trial_id,
        result.get("best_val_f05", 0.0),
        elapsed,
    )

    # Save trial result
    with open(trial_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result


def _load_pairs(path: Path) -> tuple[list[str], list[str]]:
    """Load source/target pairs from JSONL."""
    sources, targets = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            sources.append(rec["source"])
            targets.append(rec["target"])
    return sources, targets


def _train_baseline_trial(
    train_src, train_tgt, dev_src, dev_tgt,
    hparams, cfg, device, max_length, n_epochs,
    trial_dir, args,
) -> dict:
    """Train baseline for a reduced number of epochs and return val F0.5."""
    import torch
    from torch.amp import GradScaler
    from src.model.baseline import BaselineGEC
    from src.evaluation.f05_scorer import evaluate_corpus

    backbone = cfg.get("model", {}).get("pretrained", "google/byt5-small")
    model = BaselineGEC(backbone)
    model.to(device)
    model.train()

    tokenizer = model.tokenizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=hparams["lr"],
        weight_decay=cfg.get("training", {}).get("weight_decay", 0.01),
    )

    grad_accum = hparams["grad_accum_steps"]
    batch_size = hparams["batch_size"]
    use_fp16 = args.fp16 and device != "cpu"
    scaler = GradScaler(enabled=use_fp16)

    best_val_f05 = 0.0
    train_losses = []

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        optimizer.zero_grad()

        indices = list(range(len(train_src)))
        random.shuffle(indices)

        for step_i in range(0, len(indices), batch_size):
            batch_idx = indices[step_i : step_i + batch_size]
            batch_src = [train_src[i] for i in batch_idx]
            batch_tgt = [train_tgt[i] for i in batch_idx]

            enc = tokenizer(
                batch_src, return_tensors="pt", padding=True,
                truncation=True, max_length=max_length,
            ).to(device)
            dec = tokenizer(
                batch_tgt, return_tensors="pt", padding=True,
                truncation=True, max_length=max_length,
            ).to(device)

            with torch.amp.autocast("cuda", enabled=use_fp16):
                outputs = model.model(
                    input_ids=enc.input_ids,
                    attention_mask=enc.attention_mask,
                    labels=dec.input_ids,
                )
                loss = outputs.loss / grad_accum

            scaler.scale(loss).backward()
            epoch_loss += loss.item() * grad_accum
            n_batches += 1

            if n_batches % grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        avg_loss = epoch_loss / max(n_batches, 1)
        train_losses.append(avg_loss)

        # Validation
        val_f05 = _quick_val_f05(model, dev_src, dev_tgt, device, max_length)
        logger.info(
            "  Epoch %d/%d: train_loss=%.4f, val_f05=%.4f",
            epoch + 1, n_epochs, avg_loss, val_f05,
        )

        if val_f05 > best_val_f05:
            best_val_f05 = val_f05

    return {
        "model_type": "baseline",
        "best_val_f05": round(best_val_f05, 4),
        "final_train_loss": round(train_losses[-1], 4) if train_losses else None,
    }


def _train_morphaware_trial(
    train_src, train_tgt, dev_src, dev_tgt,
    hparams, cfg, device, max_length, n_epochs,
    trial_dir, args,
) -> dict:
    """Train morphaware model for a reduced number of epochs."""
    import torch
    from torch.amp import GradScaler
    from src.model.morphology_aware import MorphologyAwareGEC
    from src.morphology.analyzer import MorphologicalAnalyzer
    from src.morphology.features import FeatureExtractor
    from src.evaluation.f05_scorer import evaluate_corpus

    backbone = cfg.get("model", {}).get("pretrained", "google/byt5-small")
    analyzer = MorphologicalAnalyzer()
    feature_extractor = FeatureExtractor(analyzer)
    feature_vocab_size = feature_extractor.get_num_features()

    model = MorphologyAwareGEC(
        backbone,
        feature_vocab_size=feature_vocab_size,
        agreement_loss_weight=hparams.get("agreement_loss_weight", 0.3),
    )
    model.set_training_tools(analyzer, feature_extractor)
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=hparams["lr"],
        weight_decay=cfg.get("training", {}).get("weight_decay", 0.01),
    )

    grad_accum = hparams["grad_accum_steps"]
    batch_size = hparams["batch_size"]
    use_fp16 = args.fp16 and device != "cpu"
    scaler = GradScaler(enabled=use_fp16)

    best_val_f05 = 0.0

    for epoch in range(n_epochs):
        model.train()
        model.anneal_agreement_loss(epoch)
        epoch_loss = 0.0
        n_batches = 0
        optimizer.zero_grad()

        indices = list(range(len(train_src)))
        random.shuffle(indices)

        for step_i in range(0, len(indices), batch_size):
            batch_idx = indices[step_i : step_i + batch_size]
            batch_src = [train_src[i] for i in batch_idx]
            batch_tgt = [train_tgt[i] for i in batch_idx]

            with torch.amp.autocast("cuda", enabled=use_fp16):
                loss = model.training_step(batch_src, batch_tgt)
                loss = loss / grad_accum

            scaler.scale(loss).backward()
            epoch_loss += loss.item() * grad_accum
            n_batches += 1

            if n_batches % grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        avg_loss = epoch_loss / max(n_batches, 1)

        val_f05 = _quick_val_f05(model, dev_src, dev_tgt, device, max_length)
        logger.info(
            "  Epoch %d/%d: train_loss=%.4f, val_f05=%.4f",
            epoch + 1, n_epochs, avg_loss, val_f05,
        )

        if val_f05 > best_val_f05:
            best_val_f05 = val_f05

    return {
        "model_type": "morphaware",
        "best_val_f05": round(best_val_f05, 4),
        "agreement_loss_weight": hparams.get("agreement_loss_weight"),
    }


def _quick_val_f05(model, dev_src, dev_tgt, device, max_length) -> float:
    """Compute validation F0.5 on a small sample for speed."""
    import torch
    from src.evaluation.f05_scorer import evaluate_corpus

    model.eval()
    # Use at most 200 sentences for speed during search
    sample_n = min(200, len(dev_src))
    hyps = []
    with torch.no_grad():
        for i in range(0, sample_n, 16):
            batch = dev_src[i : i + 16]
            if hasattr(model, "correct_batch") and callable(model.correct_batch):
                hyps.extend(model.correct_batch(batch))
            else:
                for s in batch:
                    hyps.append(model.correct(s))

    metrics = evaluate_corpus(
        dev_src[:sample_n], hyps, dev_tgt[:sample_n]
    )
    return metrics.f05


def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter search for Sorani GEC models"
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--model-type",
        choices=["baseline", "morphaware"],
        default="baseline",
        help="Which model to search over",
    )
    parser.add_argument(
        "--strategy",
        choices=["grid", "random"],
        default="grid",
        help="Search strategy: grid (exhaustive) or random (budget-limited)",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=None,
        help="Max trials for random search (ignored for grid)",
    )
    parser.add_argument(
        "--search-epochs",
        type=int,
        default=5,
        help="Number of epochs per trial (reduced for speed; default: 5)",
    )
    parser.add_argument(
        "--subsample",
        type=int,
        default=None,
        help="Subsample training data to N pairs for faster search",
    )
    parser.add_argument("--output", default="results/hyperparam_search")
    parser.add_argument("--device", default=None)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)

    # Searchable hyperparameters
    parser.add_argument(
        "--lr",
        type=float,
        nargs="+",
        default=None,
        help="Learning rates to search (default: 1e-5 3e-5 5e-5 1e-4)",
    )
    parser.add_argument(
        "--batch-size-list",
        type=int,
        nargs="+",
        default=None,
        help="Batch sizes to search (default: 8 16 32)",
    )
    parser.add_argument(
        "--grad-accum-list",
        type=int,
        nargs="+",
        default=None,
        help="Grad accumulation steps to search (default: 4 8)",
    )
    parser.add_argument(
        "--agr-weights",
        type=float,
        nargs="+",
        default=None,
        help="Agreement loss weights to search (morphaware only; default: 0.1 0.3 0.5)",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    cfg = load_config(args.config) if Path(args.config).exists() else {}
    space = build_search_space(args)

    if args.strategy == "random":
        budget = args.budget or 10
        if budget < len(space):
            space = random.sample(space, budget)
            logger.info("Random search: sampled %d of %d combinations", budget, len(space))
    
    args.total_trials = len(space)
    logger.info(
        "Starting %s search: %d trials, %d epochs each, model=%s",
        args.strategy,
        len(space),
        args.search_epochs,
        args.model_type,
    )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for i, hparams in enumerate(space, 1):
        result = run_single_trial(i, hparams, args, cfg)
        results.append(result)

    # Sort by val F0.5 descending
    results.sort(key=lambda r: r.get("best_val_f05", 0.0), reverse=True)

    # Save summary
    summary = {
        "strategy": args.strategy,
        "model_type": args.model_type,
        "total_trials": len(results),
        "search_epochs": args.search_epochs,
        "subsample": args.subsample,
        "best_trial": results[0] if results else None,
        "all_trials": results,
    }
    summary_path = output_dir / "search_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("Search complete. Best trial: %s", results[0] if results else "none")
    logger.info("Summary saved to %s", summary_path)

    # Print top-5
    print("\n=== Top 5 Configurations ===")
    for i, r in enumerate(results[:5], 1):
        hp = r.get("hparams", {})
        print(
            "  %d. F0.5=%.4f | lr=%.1e batch=%d accum=%d%s | %.0fs"
            % (
                i,
                r.get("best_val_f05", 0.0),
                hp.get("lr", 0),
                hp.get("batch_size", 0),
                hp.get("grad_accum_steps", 0),
                " agr_w=%.2f" % hp["agreement_loss_weight"]
                if "agreement_loss_weight" in hp
                else "",
                r.get("elapsed_seconds", 0),
            )
        )


if __name__ == "__main__":
    main()
