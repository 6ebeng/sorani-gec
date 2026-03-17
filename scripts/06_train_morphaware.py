"""
Step 6: Train Morphology-Aware GEC Model

Extends the baseline training script with:
- Morphological feature extraction per token
- Agreement label generation for auxiliary loss
- Validation loop with early stopping
- Cosine-with-restarts LR scheduler
- Gradient accumulation
- FP16 mixed-precision training

Usage:
    python scripts/06_train_morphaware.py [--config configs/default.yaml]
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


def load_jsonl(path: Path) -> tuple[list[str], list[str], list[dict]]:
    """Load source/target pairs and metadata from JSONL."""
    sources, targets, records = [], [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            sources.append(rec["source"])
            targets.append(rec["target"])
            records.append(rec)
    return sources, targets, records


def load_plain_pairs(data_dir: Path) -> tuple[list[str], list[str]]:
    """Load source/target from .src/.tgt files (fallback for baseline format)."""
    with open(data_dir / "train.src", "r", encoding="utf-8") as f:
        sources = [l.strip() for l in f]
    with open(data_dir / "train.tgt", "r", encoding="utf-8") as f:
        targets = [l.strip() for l in f]
    return sources, targets


def main():
    parser = argparse.ArgumentParser(description="Train morphology-aware GEC model")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--model", default="google/byt5-small")
    parser.add_argument("--data-dir", default="data/splits")
    parser.add_argument("--output-dir", default="results/models/morphaware")
    parser.add_argument("--agreement-loss-weight", type=float, default=0.3)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader, Dataset
    from torch.cuda.amp import GradScaler

    from src.model.morphology_aware import MorphologyAwareGEC
    from src.morphology.analyzer import MorphologicalAnalyzer
    from src.morphology.builder import build_agreement_graph
    from src.morphology.features import FeatureExtractor
    from src.morphology.graph import EDGE_TYPE_ORDER
    from src.morphology.lexicon import SoraniLexicon

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # --- Agreement label generation (must precede model constructor) ---
    # Agreement labels match the 24 edge types in EDGE_TYPE_ORDER.
    # Class 0 = correct (no error); classes 1..24 = the corresponding edge type.
    ERROR_TYPE_MAP: dict[str, int] = {
        edge_type: idx + 1
        for idx, edge_type in enumerate(EDGE_TYPE_ORDER)
    }
    # Legacy error_type strings from JSONL metadata map to the closest edge type.
    _LEGACY_ERROR_MAP: dict[str, str] = {
        "subject_verb_number": "subject_verb",
        "noun_adjective_ezafe": "noun_det",
        "clitic_form": "clitic_agent",
        "tense_agreement": "subject_verb",
    }
    num_agreement_types = len(EDGE_TYPE_ORDER) + 1  # +1 for class 0 (correct)

    # --- Morphological analyzer ---
    lexicon = SoraniLexicon()
    analyzer = MorphologicalAnalyzer(use_klpt=False, ahmadi_lexicon=lexicon)
    feature_vocab = analyzer.build_feature_vocabulary()
    feature_vocab_size = len(feature_vocab)
    feature_extractor = FeatureExtractor(analyzer)

    # --- Load model ---
    logger.info("Loading model: %s", args.model)
    model = MorphologyAwareGEC(
        model_name=args.model,
        feature_vocab_size=feature_vocab_size,
        agreement_loss_weight=args.agreement_loss_weight,
        max_length=args.max_length,
        num_agreement_types=num_agreement_types,
    )
    model = model.to(device)

    # --- Load data ---
    data_dir = Path(args.data_dir)
    train_jsonl = data_dir / "train.jsonl"
    dev_jsonl = data_dir / "dev.jsonl"

    if train_jsonl.exists():
        train_sources, train_targets, train_records = load_jsonl(train_jsonl)
    elif (data_dir / "train.src").exists():
        train_sources, train_targets = load_plain_pairs(data_dir)
        train_records = [{} for _ in range(len(train_sources))]
    else:
        logger.error("Training data not found at %s", data_dir)
        return

    dev_sources, dev_targets = [], []
    if dev_jsonl.exists():
        dev_sources, dev_targets, _ = load_jsonl(dev_jsonl)
    logger.info("Loaded %d train, %d dev pairs", len(train_sources), len(dev_sources))

    def make_agreement_labels(source: str, record: dict, tokenizer, max_length: int) -> list[int]:
        """Generate per-byte agreement labels from error metadata."""
        encoded = tokenizer(source, max_length=max_length, truncation=True,
                            padding="max_length", return_tensors="pt")
        seq_len = encoded["input_ids"].size(1)
        labels = [0] * seq_len

        error_type = record.get("error_type", "")
        # Resolve legacy error type names to edge type names
        edge_type = _LEGACY_ERROR_MAP.get(error_type, error_type)
        label_class = ERROR_TYPE_MAP.get(edge_type, 0)
        if label_class:
            target_text = record.get("target", "")
            src_bytes = source.encode("utf-8")
            tgt_bytes = target_text.encode("utf-8")
            # Mark ALL differing byte positions (not just the first)
            for i in range(min(len(src_bytes), len(tgt_bytes))):
                if i < seq_len and src_bytes[i] != tgt_bytes[i]:
                    labels[i] = label_class
            # If lengths differ, mark trailing positions in the longer string
            if len(src_bytes) < len(tgt_bytes):
                for i in range(len(src_bytes), min(len(tgt_bytes), seq_len)):
                    labels[i] = label_class
        return labels

    # --- Dataset ---
    class MorphAwareDataset(Dataset):
        def __init__(self, sources, targets, records, tokenizer, analyzer_,
                     feature_vocab_, max_length, feature_extractor_=None):
            self.sources = sources
            self.targets = targets
            self.records = records
            self.tokenizer = tokenizer
            self.analyzer = analyzer_
            self.feature_vocab = feature_vocab_
            self.max_length = max_length
            self.feature_extractor = feature_extractor_

        def __len__(self):
            return len(self.sources)

        def __getitem__(self, idx):
            src = self.sources[idx]
            tgt = self.targets[idx]
            rec = self.records[idx]

            src_enc = self.tokenizer(src, max_length=self.max_length,
                                     truncation=True, padding="max_length",
                                     return_tensors="pt")
            tgt_enc = self.tokenizer(tgt, max_length=self.max_length,
                                     truncation=True, padding="max_length",
                                     return_tensors="pt")

            # Extract morphological features via FeatureExtractor (H3 fix)
            if self.feature_extractor is not None:
                morph_feats = self.feature_extractor.extract_features(src)
            else:
                tokens = self.analyzer.tokenize(src)
                morph_feats = []
                for tok in tokens[:self.max_length]:
                    feat = self.analyzer.analyze_token(tok)
                    indices = feat.to_vector_indices(self.feature_vocab)
                    morph_feats.append(indices)

            # Pad to max_length
            num_feat = self.feature_extractor.get_num_features() if self.feature_extractor else 9
            while len(morph_feats) < self.max_length:
                morph_feats.append([0] * num_feat)

            # Build agreement graph; use typed stacked matrix (C2: 4D, not 3D)
            graph = build_agreement_graph(src, self.analyzer)
            stacked, type_names = graph.to_typed_stacked_matrix()
            num_types = len(EDGE_TYPE_ORDER)
            n_words = len(graph.tokens)

            # Build word-to-byte mapping for alignment (H5 fix)
            # Each Kurdish character is typically 2 bytes in UTF-8.
            # The ByT5 tokenizer encodes each byte as a separate token.
            word_tokens = self.analyzer.tokenize(src)
            byte_offsets: list[tuple[int, int]] = []  # (start_byte, end_byte) per word
            cursor = 0
            src_bytes = src.encode("utf-8")
            for w in word_tokens:
                w_bytes = w.encode("utf-8")
                # Find the word boundary in the byte stream
                pos = src_bytes.find(w_bytes, cursor)
                if pos >= 0:
                    byte_offsets.append((pos, pos + len(w_bytes)))
                    cursor = pos + len(w_bytes)
                else:
                    # Fallback: use cursor position
                    byte_offsets.append((cursor, cursor + len(w_bytes)))
                    cursor += len(w_bytes)

            # Pad typed adjacency to [num_types, max_length, max_length]
            # mapped from word-level to byte-level positions
            agr_mask = [[[0] * self.max_length for _ in range(self.max_length)]
                        for _ in range(num_types)]
            for t_idx in range(min(len(stacked), num_types)):
                word_mat = stacked[t_idx]
                for r in range(min(len(word_mat), n_words)):
                    for c in range(min(len(word_mat[r]), n_words)):
                        if word_mat[r][c] == 0:
                            continue
                        # Map word indices to byte spans
                        if r < len(byte_offsets) and c < len(byte_offsets):
                            r_start, r_end = byte_offsets[r]
                            c_start, c_end = byte_offsets[c]
                            for bi in range(r_start, min(r_end, self.max_length)):
                                for bj in range(c_start, min(c_end, self.max_length)):
                                    agr_mask[t_idx][bi][bj] = 1

            # Agreement labels
            agr_labels = make_agreement_labels(src, rec, self.tokenizer,
                                               self.max_length)

            return {
                "input_ids": src_enc["input_ids"].squeeze(0),
                "attention_mask": src_enc["attention_mask"].squeeze(0),
                "labels": tgt_enc["input_ids"].squeeze(0),
                "morph_features": torch.tensor(morph_feats[:self.max_length],
                                               dtype=torch.long),
                "agreement_labels": torch.tensor(agr_labels[:self.max_length],
                                                 dtype=torch.long),
                "agreement_mask": torch.tensor(agr_mask, dtype=torch.long),
            }

    train_dataset = MorphAwareDataset(
        train_sources, train_targets, train_records,
        model.tokenizer, analyzer, feature_vocab, args.max_length,
        feature_extractor_=feature_extractor,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=0)

    dev_loader = None
    if dev_sources:
        dev_dataset = MorphAwareDataset(
            dev_sources, dev_targets, [{} for _ in range(len(dev_sources))],
            model.tokenizer, analyzer, feature_vocab, args.max_length,
            feature_extractor_=feature_extractor,
        )
        dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size,
                                shuffle=False, num_workers=0)

    # --- Optimizer + Scheduler ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    total_steps = (len(train_loader) // args.grad_accum_steps) * args.epochs
    warmup_steps = min(1000, total_steps // 10)
    from torch.optim.lr_scheduler import LinearLR, CosineAnnealingWarmRestarts, SequentialLR
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)
    cosine_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=max(1, total_steps // 3))
    scheduler = SequentialLR(optimizer, [warmup_scheduler, cosine_scheduler],
                             milestones=[warmup_steps])

    scaler = GradScaler(enabled=args.fp16 and device.type == "cuda")

    # --- Training loop ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
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
                    morph_features=batch["morph_features"],
                    labels=batch["labels"],
                    agreement_labels=batch["agreement_labels"],
                    agreement_mask=batch["agreement_mask"],
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
                logger.info("Epoch %d/%d | Batch %d | Loss: %.4f | LR: %.2e",
                            epoch + 1, args.epochs, batch_idx + 1, avg,
                            optimizer.param_groups[0]["lr"])

        avg_train_loss = total_loss / max(len(train_loader), 1)
        logger.info("Epoch %d train loss: %.4f", epoch + 1, avg_train_loss)

        # --- Validation ---
        if dev_loader:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in dev_loader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    outputs = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        morph_features=batch["morph_features"],
                        labels=batch["labels"],
                        agreement_labels=batch["agreement_labels"],
                        agreement_mask=batch["agreement_mask"],
                    )
                    val_loss += outputs["loss"].item()
            avg_val_loss = val_loss / max(len(dev_loader), 1)
            logger.info("Epoch %d val loss: %.4f", epoch + 1, avg_val_loss)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(model.state_dict(), output_dir / "best_model.pt")
                logger.info("Saved best model (val_loss=%.4f)", best_val_loss)
            else:
                patience_counter += 1
                logger.info("No improvement (%d/%d patience)", patience_counter, args.patience)
                if patience_counter >= args.patience:
                    logger.info("Early stopping triggered.")
                    break
        else:
            # No dev set — save based on train loss
            if avg_train_loss < best_val_loss:
                best_val_loss = avg_train_loss
                torch.save(model.state_dict(), output_dir / "best_model.pt")
                logger.info("Saved best model (train_loss=%.4f)", best_val_loss)

    # Save final model
    torch.save(model.state_dict(), output_dir / "final_model.pt")
    logger.info("Training complete. Best loss: %.4f", best_val_loss)


if __name__ == "__main__":
    main()
