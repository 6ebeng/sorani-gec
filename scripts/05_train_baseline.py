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
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader, Dataset
    from torch.amp import GradScaler

    from src.model.baseline import BaselineGEC

    logger.info("Device: %s", "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    logger.info("Loading model: %s", args.model)
    model = BaselineGEC(model_name=args.model)
    model = model.to(device)

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
    warmup_steps = min(1000, total_steps // 10)
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=max(1, warmup_steps))
    cosine_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=max(1, (total_steps - warmup_steps) // 3))
    scheduler = SequentialLR(optimizer, [warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])

    scaler = GradScaler("cuda", enabled=args.fp16 and device.type == "cuda")

    # Training loop
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        optimizer.zero_grad()

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
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            total_loss += loss.item() * args.grad_accum_steps

            if (batch_idx + 1) % 50 == 0:
                avg = total_loss / (batch_idx + 1)
                logger.info("Epoch %d/%d | Batch %d | Loss: %.4f",
                            epoch + 1, args.epochs, batch_idx + 1, avg)

        # Flush any remaining accumulated gradients
        if (batch_idx + 1) % args.grad_accum_steps != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        avg_train_loss = total_loss / max(len(train_loader), 1)
        logger.info("Epoch %d train loss: %.4f", epoch + 1, avg_train_loss)

        # Validation
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

            # Early stopping
            if eval_loss < best_loss:
                best_loss = eval_loss
                patience_counter = 0
                torch.save(model.state_dict(), output_dir / "best_model.pt")
                logger.info("Saved best model (loss=%.4f)", best_loss)
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
    logger.info("Training complete. Best loss: %.4f", best_loss)


if __name__ == "__main__":
    main()
