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

import yaml

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
    parser.add_argument("--augment", type=float, default=0.0,
                        help="Augmentation ratio: add N*augment augmented pairs "
                             "to training data (0 = disabled). Uses swap strategy.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (PIPE-7)")
    parser.add_argument("--selection-metric", type=str, default="val_f05",
                        choices=["val_f05", "val_loss"],
                        help="Metric for best-checkpoint and early-stopping (FM1).")
    parser.add_argument("--val-f05-subsample", type=int, default=0,
                        help="If >0, subsample the dev set to N sentences for the slow F0.5 "
                             "computation (informational only when --selection-metric=val_loss). "
                             "0 = use full dev set.")
    parser.add_argument("--num-workers", type=int, default=-1,
                        help="DataLoader worker processes for CPU-bound morphology feature "
                             "extraction. -1 = use config value (default 0).")
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
        sentinel = parser.parse_args([])  # defaults only
        for key, yaml_val in _yaml_defaults.items():
            if yaml_val is not None and getattr(args, key) == getattr(sentinel, key):
                setattr(args, key, yaml_val)
        logger.info("Loaded config from %s", cfg_path)
    else:
        logger.warning("Config file not found: %s — using CLI defaults", cfg_path)

    import math
    import random
    import torch
    from torch.utils.data import DataLoader, Dataset
    from torch.amp import GradScaler
    from torch.utils.tensorboard import SummaryWriter

    # PIPE-7: Seed all RNGs for reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    from src.model.morphology_aware import MorphologyAwareGEC
    from src.morphology.analyzer import MorphologicalAnalyzer
    from src.morphology.builder import build_agreement_graph
    from src.morphology.features import FeatureExtractor
    from src.morphology.graph import EDGE_TYPE_ORDER
    from src.morphology.lexicon import SoraniLexicon
    from src.evaluation.f05_scorer import evaluate_corpus
    from src.data.curriculum import CurriculumSampler

    if args.device:
        device = torch.device(args.device)
        logger.info("Using explicitly requested device: %s", device)
    else:
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
        feature_vocab_size=max(feature_vocab_size, 1),
        agreement_loss_weight=args.agreement_loss_weight,
        max_length=args.max_length,
        num_agreement_types=num_agreement_types,
    )
    logger.info(
        "Feature vocab: %d entries; model embedding capacity: %d",
        feature_vocab_size, max(feature_vocab_size, 1),
    )
    model = model.to(device)

    # Log parameter breakdown: backbone vs morphological components
    backbone_params = sum(p.numel() for p in model.backbone.parameters())
    morph_params = sum(p.numel() for p in model.morph_embedding.parameters())
    agr_params = sum(p.numel() for p in model.agreement_predictor.parameters())
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "Parameter count — backbone: %s, morph_embed: %s, agr_predictor: %s, total: %s",
        f"{backbone_params:,}", f"{morph_params:,}", f"{agr_params:,}", f"{total_params:,}",
    )

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

    # PIPE-8: Data augmentation (mirrors baseline 05_train_baseline.py)
    if args.augment > 0:
        from src.data.augmentation import SoraniAugmenter
        import random as _aug_rng
        augmenter = SoraniAugmenter(seed=args.seed)
        n_aug = int(len(train_sources) * args.augment)
        _aug_rand = _aug_rng.Random(args.seed)
        aug_src, aug_tgt, aug_rec = [], [], []
        for _ in range(n_aug):
            idx = _aug_rand.randint(0, len(train_sources) - 1)
            a_src, a_tgt = augmenter.augment_pair(
                train_sources[idx], train_targets[idx], strategy="swap",
            )
            aug_src.append(a_src)
            aug_tgt.append(a_tgt)
            aug_rec.append(dict(train_records[idx], augmented="swap"))
        train_sources.extend(aug_src)
        train_targets.extend(aug_tgt)
        train_records.extend(aug_rec)
        logger.info("Augmented training data: +%d pairs (%d total)",
                     n_aug, len(train_sources))

    def make_agreement_labels(source: str, record: dict, tokenizer, max_length: int) -> list[int]:
        """Generate per-byte agreement labels from error metadata.

        Uses word-level error span information when available, falling back
        to byte-level diff.  Handles insertions/deletions by capping at
        the shorter of the two byte sequences.
        """
        encoded = tokenizer(source, max_length=max_length, truncation=True,
                            padding="max_length", return_tensors="pt")
        seq_len = encoded["input_ids"].size(1)
        labels = [0] * seq_len

        error_type = record.get("error_type", "")
        # Resolve legacy error type names to edge type names
        edge_type = _LEGACY_ERROR_MAP.get(error_type, error_type)
        label_class = ERROR_TYPE_MAP.get(edge_type, 0)
        if error_type and not label_class:
            logger.warning(
                "Unknown error type '%s' (edge '%s') — agreement label "
                "falls back to 0 (no-error). This may dilute agreement loss.",
                error_type, edge_type,
            )
        if label_class:
            # Use error span positions if available (from ErrorAnnotation)
            errors = record.get("errors", [])
            if errors:
                src_bytes = source.encode("utf-8")
                for err in errors:
                    err_edge = _LEGACY_ERROR_MAP.get(err.get("type", ""), err.get("type", ""))
                    err_cls = ERROR_TYPE_MAP.get(err_edge, label_class)
                    # Convert character start/end to byte offsets
                    char_start = err.get("start", 0)
                    char_end = err.get("end", char_start)
                    if char_start >= len(source) or char_end > len(source):
                        logger.warning(
                            "Error span out of bounds: start=%d end=%d len=%d — skipped",
                            char_start, char_end, len(source),
                        )
                        continue
                    if char_start == char_end:
                        logger.debug("Zero-length error span at %d — skipped", char_start)
                        continue
                    # Validate word-boundary alignment: spans from error
                    # generators should start/end at word edges.
                    if char_start > 0 and source[char_start - 1] not in (' ', '\t', '\n') and source[char_start] not in (' ', '\t', '\n'):
                        logger.debug(
                            "Span start=%d not at word boundary (context: '…%s|%s…')",
                            char_start, source[max(0, char_start-2):char_start],
                            source[char_start:char_start+2],
                        )
                    if char_end < len(source) and source[char_end - 1] not in (' ', '\t', '\n') and source[char_end] not in (' ', '\t', '\n'):
                        logger.debug(
                            "Span end=%d not at word boundary (context: '…%s|%s…')",
                            char_end, source[max(0, char_end-2):char_end],
                            source[char_end:min(len(source), char_end+2)],
                        )
                    byte_start = len(source[:char_start].encode("utf-8"))
                    byte_end = len(source[:char_end].encode("utf-8"))
                    src_byte_len = len(src_bytes)
                    for bi in range(byte_start, min(byte_end, src_byte_len, seq_len)):
                        labels[bi] = err_cls
            else:
                # PIPE-6: difflib-based alignment (replaces naive byte diff)
                # Uses SequenceMatcher on byte sequences to correctly handle
                # insertions/deletions that shift positions.
                import difflib
                target_text = record.get("target", "")
                src_bytes = source.encode("utf-8")
                tgt_bytes = target_text.encode("utf-8")
                matcher = difflib.SequenceMatcher(
                    None, src_bytes, tgt_bytes, autojunk=False,
                )
                for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                    if tag == "equal":
                        continue
                    # Mark source-side bytes involved in replace/delete/insert
                    for bi in range(i1, min(i2, seq_len)):
                        labels[bi] = label_class
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
                morph_feats_word = self.feature_extractor.extract_features(src)
                if morph_feats_word is None:
                    logger.warning("Feature extraction returned None for: %.50s", src)
                    morph_feats_word = []
            else:
                tokens = self.analyzer.tokenize(src)
                morph_feats_word = []
                for tok in tokens[:self.max_length]:
                    feat = self.analyzer.analyze_token(tok)
                    indices = feat.to_vector_indices(self.feature_vocab)
                    morph_feats_word.append(indices)

            num_feat = self.feature_extractor.get_num_features() if self.feature_extractor else 9

            # Build agreement graph; use typed stacked matrix (C2: 4D, not 3D)
            graph = build_agreement_graph(src, self.analyzer)
            stacked, type_names = graph.to_typed_stacked_matrix()
            num_types = len(EDGE_TYPE_ORDER)
            n_words = len(graph.tokens)

            # Build word-to-byte mapping for alignment
            # Use cumulative byte tracking to avoid O(n²) re-encoding.
            word_tokens = self.analyzer.tokenize(src)
            byte_offsets: list[tuple[int, int]] = []  # (start_byte, end_byte) per word
            char_cursor = 0
            byte_cursor = 0
            for w in word_tokens:
                # Skip whitespace between words
                while char_cursor < len(src) and src[char_cursor] in (' ', '\t', '\n'):
                    byte_cursor += len(src[char_cursor].encode("utf-8"))
                    char_cursor += 1
                # Find this word starting at char_cursor
                idx = src.find(w, char_cursor)
                if idx >= 0:
                    # Advance byte_cursor past any skipped chars between char_cursor and idx
                    if idx > char_cursor:
                        byte_cursor += len(src[char_cursor:idx].encode("utf-8"))
                    byte_start = byte_cursor
                    w_bytes = len(w.encode("utf-8"))
                    byte_end = byte_start + w_bytes
                    byte_offsets.append((byte_start, byte_end))
                    char_cursor = idx + len(w)
                    byte_cursor = byte_end
                else:
                    # Fallback: estimate from cursor position
                    byte_start = byte_cursor
                    w_bytes = len(w.encode("utf-8"))
                    byte_end = byte_start + w_bytes
                    byte_offsets.append((byte_start, byte_end))
                    char_cursor += len(w)
                    byte_cursor = byte_end

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

            # Map word-level morph features to byte positions (fix 4.1)
            morph_feats = [[0] * num_feat for _ in range(self.max_length)]
            for w_idx in range(min(len(morph_feats_word), len(byte_offsets))):
                b_start, b_end = byte_offsets[w_idx]
                for bi in range(b_start, min(b_end, self.max_length)):
                    morph_feats[bi] = morph_feats_word[w_idx]

            # Agreement labels
            agr_labels = make_agreement_labels(src, rec, self.tokenizer,
                                               self.max_length)

            return {
                "input_ids": src_enc["input_ids"].squeeze(0),
                "attention_mask": src_enc["attention_mask"].squeeze(0),
                "labels": tgt_enc["input_ids"].squeeze(0),
                "morph_features": torch.tensor(morph_feats,
                                               dtype=torch.long),
                "agreement_labels": torch.tensor(agr_labels[:self.max_length],
                                                 dtype=torch.long),
                "agreement_mask": torch.tensor(agr_mask, dtype=torch.int8),
            }

    train_dataset = MorphAwareDataset(
        train_sources, train_targets, train_records,
        model.tokenizer, analyzer, feature_vocab, args.max_length,
        feature_extractor_=feature_extractor,
    )

    # Curriculum learning: sort by difficulty (word count), progressively
    # expose harder examples across epochs.
    # 6B.8: Use word count instead of character length. Arabic-script
    # characters are multi-byte in UTF-8 and inflate char-length without
    # corresponding increases in linguistic complexity.
    _nw = cfg.get("hardware", {}).get("num_workers", 0) if cfg else 0
    if args.num_workers >= 0:
        _nw = args.num_workers
    logger.info("DataLoader num_workers: %d", _nw)
    curriculum_sampler = None
    if args.curriculum:
        difficulties = [len(s.split()) for s in train_sources]
        curriculum_sampler = CurriculumSampler(
            difficulties, total_epochs=args.epochs,
        )
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  sampler=curriculum_sampler, num_workers=_nw)
        logger.info("Curriculum learning enabled (%d samples)", len(train_dataset))
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=_nw)

    dev_loader = None
    if dev_sources:
        dev_dataset = MorphAwareDataset(
            dev_sources, dev_targets, [{} for _ in range(len(dev_sources))],
            model.tokenizer, analyzer, feature_vocab, args.max_length,
            feature_extractor_=feature_extractor,
        )
        dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size,
                                shuffle=False, num_workers=_nw)

    # --- Optimizer + Scheduler ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    _t_cfg = cfg.get("training", {}) if cfg else {}
    total_steps = math.ceil(len(train_loader) / args.grad_accum_steps) * args.epochs
    warmup_steps = _t_cfg.get("warmup_steps", min(1000, total_steps // 10))
    from torch.optim.lr_scheduler import LinearLR, CosineAnnealingWarmRestarts, SequentialLR
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)
    _cosine_restarts = _t_cfg.get("cosine_restarts", 3)
    cosine_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=max(1, total_steps // _cosine_restarts))
    scheduler = SequentialLR(optimizer, [warmup_scheduler, cosine_scheduler],
                             milestones=[warmup_steps])

    scaler = GradScaler("cuda", enabled=args.fp16 and device.type == "cuda")

    # --- Training loop ---
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
    saved_checkpoints: list[tuple[float, Path]] = []
    sel_metric = args.selection_metric
    logger.info("Checkpoint selection metric: %s", sel_metric)

    def save_checkpoint(model_, optimizer_, scheduler_, scaler_, epoch_: int,
                        global_step_: int, best_f05_: float, val_f05: float,
                        val_loss_: float, tag: str) -> None:
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
            "feature_vocab_size": feature_vocab_size,
        }, ckpt_path)
        saved_checkpoints.append((score, ckpt_path))
        saved_checkpoints.sort(key=lambda x: x[0], reverse=True)
        while len(saved_checkpoints) > top_k:
            _, evicted = saved_checkpoints.pop()
            if evicted.exists():
                evicted.unlink()
                logger.info("Evicted checkpoint: %s", evicted.name)
        best_path = output_dir / "best_model.pt"
        if saved_checkpoints:
            import shutil
            shutil.copy2(str(saved_checkpoints[0][1]), str(best_path))

    def run_validation():
        """Evaluate on dev set; returns average val loss."""
        model.eval()
        val_loss_ = 0
        with torch.no_grad():
            for vb in dev_loader:
                vb = {k: v.to(device) for k, v in vb.items()}
                vo = model(
                    input_ids=vb["input_ids"],
                    attention_mask=vb["attention_mask"],
                    morph_features=vb["morph_features"],
                    labels=vb["labels"],
                    agreement_labels=vb["agreement_labels"],
                    agreement_mask=vb["agreement_mask"],
                )
                val_loss_ += vo["loss"].item()
        return val_loss_ / max(len(dev_loader), 1)

    def compute_val_f05():
        """Generate corrections on dev set using morphology and compute F0.5.

        When `--val-f05-subsample N` is set (N>0) we evaluate F0.5 on a deterministic
        N-sentence subsample of the dev set. F0.5 here is informational when
        selection-metric=val_loss, so a fixed slice is fine.
        """
        model.eval()
        if args.val_f05_subsample and args.val_f05_subsample > 0:
            n = min(args.val_f05_subsample, len(dev_sources))
            srcs_eval = dev_sources[:n]
            tgts_eval = dev_targets[:n]
        else:
            srcs_eval = dev_sources
            tgts_eval = dev_targets
        # Silence per-token morphology spam during eval to avoid I/O bottleneck.
        _builder_log = logging.getLogger("src.morphology.builder")
        _prev_level = _builder_log.level
        _builder_log.setLevel(logging.ERROR)
        hypotheses = []
        try:
            with torch.no_grad():
                _beam_w = cfg.get("evaluation", {}).get("beam_width", 4) if cfg else 4
                for src in srcs_eval:
                    try:
                        hyp = model.correct_with_morphology(
                            src, analyzer, feature_extractor, num_beams=_beam_w
                        )
                    except Exception:
                        logger.warning("correct_with_morphology() failed for: %.50s", src)
                        hyp = src  # fallback: return source unchanged
                    hypotheses.append(hyp)
        finally:
            _builder_log.setLevel(_prev_level)
        metrics = evaluate_corpus(srcs_eval, hypotheses, tgts_eval)
        return metrics.f05, metrics

    best_score = 0.0 if sel_metric == "val_f05" else -float("inf")
    best_f05 = 0.0  # legacy slot for resume-compat
    patience_counter = 0
    global_step = 0
    start_epoch = 0

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
            best_f05, _ = compute_val_f05()
            logger.info("Re-evaluated resumed checkpoint: F0.5=%.4f", best_f05)
            best_score = best_f05 if sel_metric == "val_f05" else -float("inf")
        else:
            best_f05 = ckpt.get("best_f05", 0.0)
            best_score = best_f05 if sel_metric == "val_f05" else -float("inf")
        patience_counter = ckpt.get("patience_counter", 0)
        logger.info("Resumed from %s (epoch %d, step %d, best_f05=%.4f, patience=%d)",
                     args.resume_from, start_epoch, global_step, best_f05, patience_counter)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0
        optimizer.zero_grad()

        # PIPE-14: Anneal agreement loss weight (warmup from config)
        agr_warmup = cfg.get("training", {}).get("agreement_warmup_epochs", 5) if cfg else 5
        agr_weight = model.anneal_agreement_loss(epoch, warmup_epochs=agr_warmup)
        logger.info("Epoch %d — agreement_loss_weight: %.4f", epoch + 1, agr_weight)

        if curriculum_sampler is not None:
            curriculum_sampler.set_epoch(epoch)

        batch_idx = -1
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
                tb_writer.add_scalar("train/agreement_loss", outputs["agreement_loss"].item(), global_step)
                tb_writer.add_scalar("train/seq2seq_loss", outputs["seq2seq_loss"].item(), global_step)
                tb_writer.add_scalar("train/agreement_weight", model.agreement_loss_weight, global_step)

                # Intra-epoch eval every N optimizer steps
                if (args.eval_every_n_steps > 0
                        and dev_loader
                        and global_step % args.eval_every_n_steps == 0):
                    step_val_loss = run_validation()
                    logger.info("Step %d val loss: %.4f", global_step, step_val_loss)
                    tb_writer.add_scalar("val/loss", step_val_loss, global_step)
                    model.train()

            total_loss += loss.item() * args.grad_accum_steps

            _log_every = _t_cfg.get("log_every_n_steps", 50)
            if (batch_idx + 1) % _log_every == 0:
                avg = total_loss / (batch_idx + 1)
                logger.info("Epoch %d/%d | Batch %d | Loss: %.4f | LR: %.2e",
                            epoch + 1, args.epochs, batch_idx + 1, avg,
                            optimizer.param_groups[0]["lr"])

        # Flush any remaining accumulated gradients
        if batch_idx >= 0 and (batch_idx + 1) % args.grad_accum_steps != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), _clip_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        avg_train_loss = total_loss / max(len(train_loader), 1)
        logger.info("Epoch %d train loss: %.4f", epoch + 1, avg_train_loss)
        tb_writer.add_scalar("epoch/train_loss", avg_train_loss, epoch + 1)

        # --- Validation ---
        if dev_loader:
            avg_val_loss = run_validation()
            logger.info("Epoch %d val loss: %.4f", epoch + 1, avg_val_loss)
            tb_writer.add_scalar("epoch/val_loss", avg_val_loss, epoch + 1)

            # Compute F0.5 on dev set for model selection
            val_f05, val_metrics = compute_val_f05()
            logger.info("Epoch %d val F0.5: %.4f (%s)", epoch + 1, val_f05, val_metrics)
            tb_writer.add_scalar("epoch/val_f05", val_f05, epoch + 1)
            tb_writer.add_scalar("epoch/val_precision", val_metrics.precision, epoch + 1)
            tb_writer.add_scalar("epoch/val_recall", val_metrics.recall, epoch + 1)

            # Log to CSV
            csv_writer.writerow([
                epoch + 1, f"{avg_train_loss:.6f}", f"{avg_val_loss:.6f}",
                f"{val_f05:.6f}", f"{val_metrics.precision:.6f}", f"{val_metrics.recall:.6f}",
                f"{optimizer.param_groups[0]['lr']:.8f}",
            ])
            csv_log_file.flush()

            save_checkpoint(model, optimizer, scheduler, scaler, epoch,
                            global_step, best_score, val_f05, avg_val_loss, f"epoch{epoch + 1}")
            current = val_f05 if sel_metric == "val_f05" else -avg_val_loss
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
            # No dev set — early stopping disabled; save latest model
            logger.warning("No dev set — early stopping disabled; saving latest model")
            torch.save(model.state_dict(), output_dir / "best_model.pt")

    # Save final model
    torch.save(model.state_dict(), output_dir / "final_model.pt")
    csv_log_file.close()
    tb_writer.close()
    logger.info("Training complete. Best %s score: %.4f", sel_metric, best_score)
    logger.info("Training log saved to %s", csv_log_path)


if __name__ == "__main__":
    main()
