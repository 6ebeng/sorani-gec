"""
Step 5: Train Baseline GEC Model

Features:
- ByT5-small backbone
- Validation loop with early stopping (patience configurable)
- Cosine-with-restarts LR scheduler
- Gradient accumulation
- FP16 mixed-precision training

Usage:
    python scripts/05_train_baseline.py [--config configs/default.yaml]
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_pairs(path: Path) -> tuple[list[str], list[str]]:
    """Load pairs from .src/.tgt or .jsonl format."""
    if path.suffix == ".jsonl" and path.exists():
        sources, targets = [], []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line.strip())
                sources.append(rec["source"])
                targets.append(rec["target"])
        return sources, targets

    src_path = path.parent / (path.stem + ".src")
    tgt_path = path.parent / (path.stem + ".tgt")
    with open(src_path, "r", encoding="utf-8") as f:
        sources = [l.strip() for l in f]
    with open(tgt_path, "r", encoding="utf-8") as f:
        targets = [l.strip() for l in f]
    return sources, targets


def main():
    parser = argparse.ArgumentParser(description="Train baseline GEC model")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--model", default="google/byt5-small")
    parser.add_argument("--data-dir", default="data/splits")
    parser.add_argument("--output-dir", default="results/models/baseline")
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--save-top-k", type=int, default=3,
                        help="Keep the K best checkpoints by val loss")
    parser.add_argument("--eval-every-n-steps", type=int, default=500,
                        help="Run validation every N optimizer steps (0=epoch-only)")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Path to checkpoint .pt file to resume training from")
    parser.add_argument("--device", type=str, default=None,
                        help="Explicitly set a device (e.g., 'cuda:0', 'cpu').")
    parser.add_argument("--curriculum", action="store_true", default=False,
                        help="Enable curriculum learning (easy→hard by sentence length)")
    parser.add_argument("--curriculum-morphology", action="store_true", default=False,
                        help="Use morphology-aware difficulty (edge count + word count) "
                             "instead of word count only. Requires --curriculum.")
    parser.add_argument("--augment", type=float, default=0.0,
                        help="Augmentation ratio: add N*augment augmented pairs "
                             "to training data (0 = disabled). Uses swap strategy.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (PIPE-7)")
    parser.add_argument("--selection-metric", type=str, default="val_f05",
                        choices=["val_f05", "val_loss"],
                        help="Metric for best-checkpoint and early-stopping (FM1).")
    args = parser.parse_args()
    cfg = {}

    # Load YAML config and use as defaults; CLI args override.
    cfg_path = Path(args.config)
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
        training_cfg = cfg.get("training", {})
        _yaml_defaults = {
            "max_length": cfg.get("data", {}).get("max_seq_length"),
            "batch_size": training_cfg.get("batch_size"),
            "grad_accum_steps": training_cfg.get("gradient_accumulation_steps"),
            "fp16": training_cfg.get("fp16"),
            "lr": training_cfg.get("learning_rate"),
            "patience": training_cfg.get("early_stopping_patience"),
        }
        # Only apply YAML value when the user did not pass the flag on CLI.
        sentinel = parser.parse_args([])  # defaults only
        for key, yaml_val in _yaml_defaults.items():
            if yaml_val is not None and getattr(args, key) == getattr(sentinel, key):
                setattr(args, key, yaml_val)
        logger.info("Loaded config from %s", cfg_path)
    else:
        logger.warning("Config file not found: %s — using CLI defaults", cfg_path)

    import random
    import torch
    from torch.utils.data import DataLoader, Dataset
    from torch.amp import GradScaler
    from torch.utils.tensorboard import SummaryWriter

    from src.model.baseline import BaselineGEC
    from src.evaluation.f05_scorer import evaluate_corpus
    from src.data.curriculum import CurriculumSampler

    # PIPE-7: Seed all RNGs for reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    logger.info("Auto-Device: %s", "cuda" if torch.cuda.is_available() else "cpu")
    if args.device:
        device = torch.device(args.device)
        logger.info("Using explicitly requested device: %s", device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    logger.info("Loading model: %s", args.model)
    model = BaselineGEC(model_name=args.model)
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Total parameters: %s", f"{total_params:,}")

    # Load data
    data_dir = Path(args.data_dir)
    train_jsonl = data_dir / "train.jsonl"
    dev_jsonl = data_dir / "dev.jsonl"

    if train_jsonl.exists():
        train_sources, train_targets = load_pairs(train_jsonl)
    elif (data_dir / "train.src").exists():
        train_sources, train_targets = load_pairs(data_dir / "train.src")
    else:
        logger.error("Training data not found at %s", data_dir)
        return

    dev_sources, dev_targets = [], []
    if dev_jsonl.exists():
        dev_sources, dev_targets = load_pairs(dev_jsonl)
    elif (data_dir / "dev.src").exists():
        dev_sources, dev_targets = load_pairs(data_dir / "dev.src")
    logger.info("Loaded %d train, %d dev pairs", len(train_sources), len(dev_sources))

    # PIPE-12: Data augmentation
    if args.augment > 0:
        from src.data.augmentation import SoraniAugmenter
        augmenter = SoraniAugmenter(seed=args.seed)
        n_aug = int(len(train_sources) * args.augment)
        import random as _aug_rng
        _aug_rand = _aug_rng.Random(args.seed)
        aug_src, aug_tgt = [], []
        for _ in range(n_aug):
            idx = _aug_rand.randint(0, len(train_sources) - 1)
            a_src, a_tgt = augmenter.augment_pair(
                train_sources[idx], train_targets[idx], strategy="swap",
            )
            aug_src.append(a_src)
            aug_tgt.append(a_tgt)
        train_sources.extend(aug_src)
        train_targets.extend(aug_tgt)
        logger.info("Augmented training data: +%d pairs (%d total)",
                     n_aug, len(train_sources))

    class GECDataset(Dataset):
        def __init__(self, sources, targets, tokenizer, max_length=128):
            self.sources = sources
            self.targets = targets
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.sources)

        def __getitem__(self, idx):
            src_enc = self.tokenizer(
                self.sources[idx], max_length=self.max_length,
                truncation=True, padding="max_length", return_tensors="pt",
            )
            tgt_enc = self.tokenizer(
                self.targets[idx], max_length=self.max_length,
                truncation=True, padding="max_length", return_tensors="pt",
            )
            return {
                "input_ids": src_enc["input_ids"].squeeze(0),
                "attention_mask": src_enc["attention_mask"].squeeze(0),
                "labels": tgt_enc["input_ids"].squeeze(0),
            }

    train_dataset = GECDataset(train_sources, train_targets, model.tokenizer, args.max_length)

    # Curriculum learning: sort by difficulty (word count), progressively
    # expose harder examples across epochs.
    # 6B.8: Use word count instead of character length.
    curriculum_sampler = None
    if args.curriculum:
        # PIPE-11: morphology-aware difficulty when requested
        if args.curriculum_morphology:
            from src.data.curriculum import compute_morphology_difficulty
            from src.morphology.analyzer import MorphologicalAnalyzer
            _cur_analyzer = MorphologicalAnalyzer(use_klpt=False)
            _edge_w = cfg.get("training", {}).get("curriculum_edge_weight", 0.5) if cfg else 0.5
            difficulties = compute_morphology_difficulty(
                train_sources, analyzer=_cur_analyzer, edge_weight=_edge_w,
            )
            logger.info("Curriculum: morphology-aware difficulty scores computed")
        else:
            difficulties = [len(s.split()) for s in train_sources]
        curriculum_sampler = CurriculumSampler(
            difficulties, total_epochs=args.epochs,
        )
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  sampler=curriculum_sampler)
        logger.info("Curriculum learning enabled (%d samples)", len(train_dataset))
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    dev_loader = None
    if dev_sources:
        dev_dataset = GECDataset(dev_sources, dev_targets, model.tokenizer, args.max_length)
        dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False)

    # Optimizer + Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    import math
    total_steps = math.ceil(len(train_loader) / args.grad_accum_steps) * args.epochs
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR
    _t_cfg = cfg.get("training", {}) if cfg else {}
    warmup_steps = _t_cfg.get("warmup_steps", min(1000, total_steps // 10))
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=max(1, warmup_steps))
    _cosine_restarts = _t_cfg.get("cosine_restarts", 3)
    cosine_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=max(1, (total_steps - warmup_steps) // _cosine_restarts))
    scheduler = SequentialLR(optimizer, [warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])

    scaler = GradScaler("cuda", enabled=args.fp16 and device.type == "cuda")

    # Defaults — must precede resume block so checkpoint can override
    sel_metric_init = args.selection_metric
    best_score = 0.0 if sel_metric_init == "val_f05" else -float("inf")
    best_f05 = 0.0  # legacy, retained for resume-compat
    patience_counter = 0
    global_step = 0  # counts optimizer steps
    start_epoch = 0

    def compute_val_f05(model_, dev_srcs, dev_tgts):
        """Generate corrections on dev set and compute F0.5."""
        model_.eval()
        hypotheses = []
        with torch.no_grad():
            for i in range(0, len(dev_srcs), args.batch_size):
                batch_src = dev_srcs[i:i + args.batch_size]
                _beam_w = cfg.get("evaluation", {}).get("beam_width", 4) if cfg else 4
                hypotheses.extend(model_.correct_batch(batch_src, num_beams=_beam_w))
        metrics = evaluate_corpus(dev_srcs, hypotheses, dev_tgts)
        return metrics.f05, metrics

    # Resume from checkpoint if requested (ARCH-4)
    if args.resume_from:
        ckpt = torch.load(args.resume_from, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["global_step"]
        # Re-evaluate to get current best score from actual model state
        if dev_loader and dev_sources:
            best_f05, _ = compute_val_f05(model, dev_sources, dev_targets)
            logger.info("Re-evaluated resumed checkpoint: F0.5=%.4f", best_f05)
            best_score = best_f05 if sel_metric_init == "val_f05" else -float("inf")
        else:
            best_f05 = ckpt.get("best_f05", 0.0)
            best_score = best_f05 if sel_metric_init == "val_f05" else -float("inf")
        patience_counter = ckpt.get("patience_counter", 0)
        logger.info("Resumed from %s (epoch %d, step %d, best_f05=%.4f, patience=%d)",
                     args.resume_from, start_epoch, global_step, best_f05, patience_counter)

    # Training loop
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # TensorBoard writer
    tb_writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))

    # CSV training log for easy downstream plotting
    import csv
    csv_log_path = output_dir / "training_log.csv"
    csv_log_file = open(csv_log_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_log_file)
    csv_writer.writerow(["epoch", "train_loss", "val_loss", "val_f05", "val_precision", "val_recall", "lr"])

    import atexit
    atexit.register(csv_log_file.close)
    atexit.register(tb_writer.close)

    # Checkpoint manager — keeps top-K checkpoints ranked by selection metric (higher=better)
    top_k = args.save_top_k
    # List of (score, path) sorted descending by score (best first)
    saved_checkpoints: list[tuple[float, Path]] = []
    sel_metric = sel_metric_init
    logger.info("Checkpoint selection metric: %s", sel_metric)

    def save_checkpoint(model_, optimizer_, scheduler_, scaler_, epoch_: int,
                        global_step_: int, best_f05_: float, val_f05: float,
                        val_loss_: float, tag: str) -> None:
        """Save a full training checkpoint and evict worst if exceeding top_k."""
        # Score: higher is better. For val_loss, negate so smaller loss => higher score.
        score = val_f05 if sel_metric == "val_f05" else -val_loss_
        ckpt_path = output_dir / f"checkpoint_{tag}_f05{val_f05:.4f}_loss{val_loss_:.4f}.pt"
        torch.save({
            "model_state_dict": model_.state_dict(),
            "optimizer_state_dict": optimizer_.state_dict(),
            "scheduler_state_dict": scheduler_.state_dict(),
            "scaler_state_dict": scaler_.state_dict(),
            "epoch": epoch_,
            "global_step": global_step_,
            "best_f05": best_f05_,
            "patience_counter": patience_counter,
        }, ckpt_path)
        saved_checkpoints.append((score, ckpt_path))
        saved_checkpoints.sort(key=lambda x: x[0], reverse=True)  # best score first
        while len(saved_checkpoints) > top_k:
            _, evicted = saved_checkpoints.pop()  # worst (lowest F0.5)
            if evicted.exists():
                evicted.unlink()
                logger.info("Evicted checkpoint: %s", evicted.name)
        # Copy best to best_model.pt
        best_path = output_dir / "best_model.pt"
        if saved_checkpoints:
            import shutil
            shutil.copy2(str(saved_checkpoints[0][1]), str(best_path))

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0
        batch_idx = -1
        optimizer.zero_grad()
        if curriculum_sampler is not None:
            curriculum_sampler.set_epoch(epoch)

        for batch_idx, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.autocast(device_type=device.type, enabled=args.fp16):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = outputs["loss"] / args.grad_accum_steps

            scaler.scale(loss).backward()

            if (batch_idx + 1) % args.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                _clip_norm = cfg.get("training", {}).get("gradient_clip_norm", 1.0) if cfg else 1.0
                torch.nn.utils.clip_grad_norm_(model.parameters(), _clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

                # Log training loss and LR to TensorBoard
                tb_writer.add_scalar("train/loss", loss.item() * args.grad_accum_steps, global_step)
                tb_writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)

                # Intra-epoch eval every N optimizer steps
                if (args.eval_every_n_steps > 0
                        and dev_loader
                        and global_step % args.eval_every_n_steps == 0):
                    model.eval()
                    step_val_loss = 0
                    with torch.no_grad():
                        for vb in dev_loader:
                            vb = {k: v.to(device) for k, v in vb.items()}
                            vo = model(
                                input_ids=vb["input_ids"],
                                attention_mask=vb["attention_mask"],
                                labels=vb["labels"],
                            )
                            step_val_loss += vo["loss"].item()
                    step_val_loss /= max(len(dev_loader), 1)
                    logger.info("Step %d val loss: %.4f", global_step, step_val_loss)
                    tb_writer.add_scalar("val/loss", step_val_loss, global_step)
                    model.train()

            total_loss += loss.item() * args.grad_accum_steps

            _log_every = cfg.get("training", {}).get("log_every_n_steps", 50) if cfg else 50
            if (batch_idx + 1) % _log_every == 0:
                avg = total_loss / (batch_idx + 1)
                logger.info("Epoch %d/%d | Batch %d | Loss: %.4f",
                            epoch + 1, args.epochs, batch_idx + 1, avg)

        # Flush any remaining accumulated gradients
        if batch_idx >= 0 and (batch_idx + 1) % args.grad_accum_steps != 0:
            scaler.unscale_(optimizer)
            _clip_norm = cfg.get("training", {}).get("gradient_clip_norm", 1.0) if cfg else 1.0
            torch.nn.utils.clip_grad_norm_(model.parameters(), _clip_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        avg_train_loss = total_loss / max(len(train_loader), 1)
        logger.info("Epoch %d train loss: %.4f", epoch + 1, avg_train_loss)
        tb_writer.add_scalar("epoch/train_loss", avg_train_loss, epoch + 1)

        # Validation — use F0.5 (task metric) for model selection
        if dev_loader:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in dev_loader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    outputs = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                    )
                    val_loss += outputs["loss"].item()
            eval_loss = val_loss / max(len(dev_loader), 1)
            logger.info("Epoch %d val loss: %.4f", epoch + 1, eval_loss)
            tb_writer.add_scalar("epoch/val_loss", eval_loss, epoch + 1)

            # Compute F0.5 on dev set for model selection
            val_f05, val_metrics = compute_val_f05(model, dev_sources, dev_targets)
            logger.info("Epoch %d val F0.5: %.4f (%s)", epoch + 1, val_f05, val_metrics)
            tb_writer.add_scalar("epoch/val_f05", val_f05, epoch + 1)
            tb_writer.add_scalar("epoch/val_precision", val_metrics.precision, epoch + 1)
            tb_writer.add_scalar("epoch/val_recall", val_metrics.recall, epoch + 1)

            # Log to CSV
            csv_writer.writerow([
                epoch + 1, f"{avg_train_loss:.6f}", f"{eval_loss:.6f}",
                f"{val_f05:.6f}", f"{val_metrics.precision:.6f}", f"{val_metrics.recall:.6f}",
                f"{optimizer.param_groups[0]['lr']:.8f}",
            ])
            csv_log_file.flush()

            # Early stopping + top-K checkpointing by selection metric
            save_checkpoint(model, optimizer, scheduler, scaler, epoch,
                            global_step, best_score, val_f05, eval_loss, f"epoch{epoch + 1}")
            current = val_f05 if sel_metric == "val_f05" else -eval_loss
            if current > best_score:
                best_score = current
                patience_counter = 0
                logger.info("Saved best model (%s score=%.4f)", sel_metric, best_score)
            else:
                patience_counter += 1
                logger.info("No improvement (%d/%d patience)", patience_counter, args.patience)
                if patience_counter >= args.patience:
                    logger.info("Early stopping triggered.")
                    break
        else:
            logger.warning("No dev set — early stopping disabled; saving latest model")
            torch.save(model.state_dict(), output_dir / "best_model.pt")

    torch.save(model.state_dict(), output_dir / "final_model.pt")
    csv_log_file.close()
    tb_writer.close()
    logger.info("Training complete. Best %s score: %.4f", sel_metric, best_score)
    logger.info("Training log saved to %s", csv_log_path)


if __name__ == "__main__":
    main()
