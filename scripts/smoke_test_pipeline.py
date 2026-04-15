"""
Smoke Test: End-to-End Pipeline Verification

Runs a tiny training cycle for both Baseline and MorphologyAware models
on the smoke_splits data, then evaluates with F0.5 and agreement accuracy.
Designed to complete in ~5-10 minutes on a CPU laptop.

Usage:
    python scripts/smoke_test_pipeline.py [--config configs/laptop_smoke.yaml]
"""

import argparse
import json
import logging
import math
import os
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_pairs(path: Path) -> tuple[list[str], list[str]]:
    """Load (source, target) pairs from .jsonl format."""
    sources, targets = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line.strip())
            sources.append(rec["source"])
            targets.append(rec["target"])
    return sources, targets


def timed(label: str):
    """Simple context-manager timer for logging."""
    class _Timer:
        def __enter__(self):
            self.t0 = time.perf_counter()
            return self
        def __exit__(self, *_):
            elapsed = time.perf_counter() - self.t0
            logger.info("%s completed in %.1f s", label, elapsed)
    return _Timer()


# ---------------------------------------------------------------------------
# Phase 1 — Data verification
# ---------------------------------------------------------------------------

def verify_data(data_dir: Path, max_train: int = 8, max_dev: int = 4, max_test: int = 4) -> tuple[list[str], list[str], list[str], list[str], list[str], list[str]]:
    """Load and sanity-check smoke split data (truncated for laptop speed)."""
    logger.info("=== Phase 1: Data verification ===")
    train_src, train_tgt = load_pairs(data_dir / "train.jsonl")
    dev_src, dev_tgt = load_pairs(data_dir / "dev.jsonl")
    test_src, test_tgt = load_pairs(data_dir / "test.jsonl")

    logger.info("Full data — Train: %d | Dev: %d | Test: %d",
                len(train_src), len(dev_src), len(test_src))

    # Truncate for laptop speed
    train_src, train_tgt = train_src[:max_train], train_tgt[:max_train]
    dev_src, dev_tgt = dev_src[:max_dev], dev_tgt[:max_dev]
    test_src, test_tgt = test_src[:max_test], test_tgt[:max_test]
    logger.info("Truncated — Train: %d | Dev: %d | Test: %d",
                len(train_src), len(dev_src), len(test_src))

    assert len(train_src) > 0, "Training data is empty"
    assert len(dev_src) > 0, "Dev data is empty"
    assert len(test_src) > 0, "Test data is empty"
    assert len(train_src) == len(train_tgt), "Train src/tgt length mismatch"

    # Verify encoding — check that Kurdish text is present
    sample = train_src[0]
    assert any(0x0600 <= ord(c) <= 0x06FF or 0xFB50 <= ord(c) <= 0xFDFF for c in sample), \
        "Sample does not contain Arabic-script characters: %s" % sample[:50]

    logger.info("Data verification passed. Sample: %s", sample[:80])
    return train_src, train_tgt, dev_src, dev_tgt, test_src, test_tgt


# ---------------------------------------------------------------------------
# Phase 2 — Error pipeline verification
# ---------------------------------------------------------------------------

def verify_error_pipeline(test_sentences: list[str]) -> None:
    """Run the error pipeline on a few sentences to verify generators work."""
    logger.info("=== Phase 2: Error pipeline verification ===")
    from src.errors.pipeline import ErrorPipeline

    pipeline = ErrorPipeline(error_rate=0.3, seed=42)
    logger.info("Initialized pipeline with %d generators", len(pipeline.generators))

    n_tested = 0
    n_with_errors = 0
    for sent in test_sentences[:10]:
        result = pipeline.process_sentence(sent)
        if result.has_errors:
            n_with_errors += 1
            logger.info("  Error injected: %s -> %s (%d errors)",
                        sent[:40], result.corrupted[:40], len(result.errors))
        n_tested += 1

    logger.info("Error pipeline: %d/%d sentences had errors injected", n_with_errors, n_tested)


# ---------------------------------------------------------------------------
# Phase 3 — Morphology & feature extraction verification
# ---------------------------------------------------------------------------

def verify_morphology(test_sentences: list[str]) -> None:
    """Verify morphological analyzer and feature extractor work."""
    logger.info("=== Phase 3: Morphology verification ===")
    from src.morphology.analyzer import MorphologicalAnalyzer
    from src.morphology.features import FeatureExtractor
    from src.morphology.builder import build_agreement_graph

    analyzer = MorphologicalAnalyzer(use_klpt=False)
    extractor = FeatureExtractor(analyzer=analyzer)

    for sent in test_sentences[:5]:
        # Analyzer
        tokens = analyzer.tokenize(sent)
        analyses = [analyzer.analyze_token(t) for t in tokens[:3]]
        logger.info("  Tokens (%d): %s", len(tokens), " ".join(tokens[:5]))

        # Feature extraction
        features = extractor.extract_features(sent)
        logger.info("  Features: %d words, %d dims each",
                    len(features), len(features[0]) if features else 0)

        # Agreement graph
        graph = build_agreement_graph(sent, analyzer)
        logger.info("  Agreement graph: %d tokens, %d edges",
                    len(graph.tokens), len(graph.edges))

    logger.info("Morphology verification passed")


# ---------------------------------------------------------------------------
# Phase 4 — Baseline model training
# ---------------------------------------------------------------------------

def train_baseline(
    train_src: list[str], train_tgt: list[str],
    dev_src: list[str], dev_tgt: list[str],
    cfg: dict, output_dir: Path,
) -> float:
    """Train baseline ByT5 model for a few epochs and return best val F0.5."""
    logger.info("=== Phase 4: Baseline training ===")
    import torch
    from torch.utils.data import DataLoader, Dataset
    from src.model.baseline import BaselineGEC
    from src.evaluation.f05_scorer import evaluate_corpus

    t_cfg = cfg.get("training", {})
    batch_size = t_cfg.get("batch_size", 4)
    max_length = cfg.get("data", {}).get("max_seq_length", 64)
    epochs = t_cfg.get("max_epochs", 2)
    lr = t_cfg.get("learning_rate", 5e-5)
    grad_accum = t_cfg.get("gradient_accumulation_steps", 1)
    beam_width = cfg.get("evaluation", {}).get("beam_width", 2)

    device = torch.device("cpu")
    logger.info("Device: %s", device)

    model = BaselineGEC(
        model_name=cfg.get("model", {}).get("pretrained", "google/byt5-small"),
        max_length=max_length,
    )
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Baseline params: %s", f"{total_params:,}")

    class _Dataset(Dataset):
        def __init__(self, srcs, tgts, tokenizer, ml):
            self.srcs, self.tgts = srcs, tgts
            self.tokenizer, self.ml = tokenizer, ml
        def __len__(self):
            return len(self.srcs)
        def __getitem__(self, idx):
            s = self.tokenizer(self.srcs[idx], max_length=self.ml,
                               truncation=True, padding="max_length", return_tensors="pt")
            t = self.tokenizer(self.tgts[idx], max_length=self.ml,
                               truncation=True, padding="max_length", return_tensors="pt")
            return {
                "input_ids": s["input_ids"].squeeze(0),
                "attention_mask": s["attention_mask"].squeeze(0),
                "labels": t["input_ids"].squeeze(0),
            }

    train_ds = _Dataset(train_src, train_tgt, model.tokenizer, max_length)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    dev_ds = _Dataset(dev_src, dev_tgt, model.tokenizer, max_length)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = math.ceil(len(train_loader) / grad_accum) * epochs
    warmup_steps = t_cfg.get("warmup_steps", 5)
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR
    warmup_sched = LinearLR(optimizer, start_factor=0.1, total_iters=max(1, warmup_steps))
    cosine_sched = CosineAnnealingWarmRestarts(optimizer, T_0=max(1, total_steps - warmup_steps))
    scheduler = SequentialLR(optimizer, [warmup_sched, cosine_sched], milestones=[warmup_steps])

    best_f05 = 0.0
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = outputs["loss"] / grad_accum
            loss.backward()

            if (batch_idx + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            total_loss += outputs["loss"].item()
            logger.info("  [Baseline] Epoch %d Batch %d/%d — loss: %.4f",
                        epoch + 1, batch_idx + 1, len(train_loader), outputs["loss"].item())

        avg_loss = total_loss / max(len(train_loader), 1)
        logger.info("Epoch %d/%d — train loss: %.4f", epoch + 1, epochs, avg_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in dev_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                o = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                val_loss += o["loss"].item()
        val_loss /= max(len(dev_loader), 1)

        # F0.5 on dev
        hypotheses = model.correct_batch(dev_src, num_beams=beam_width)
        metrics = evaluate_corpus(dev_src, hypotheses, dev_tgt)
        logger.info("Epoch %d — val loss: %.4f | %s", epoch + 1, val_loss, metrics)

        if metrics.f05 >= best_f05:
            best_f05 = metrics.f05
            torch.save(model.state_dict(), output_dir / "best_model.pt")

    torch.save(model.state_dict(), output_dir / "final_model.pt")
    logger.info("Baseline training done. Best F0.5: %.4f", best_f05)
    return best_f05


# ---------------------------------------------------------------------------
# Phase 5 — Morphology-aware model training
# ---------------------------------------------------------------------------

def train_morphaware(
    train_src: list[str], train_tgt: list[str],
    dev_src: list[str], dev_tgt: list[str],
    cfg: dict, output_dir: Path,
) -> float:
    """Train morphology-aware model for a few epochs."""
    logger.info("=== Phase 5: Morphology-aware training ===")
    import torch
    from torch.utils.data import DataLoader, Dataset
    from src.model.morphology_aware import MorphologyAwareGEC
    from src.morphology.analyzer import MorphologicalAnalyzer
    from src.morphology.features import FeatureExtractor
    from src.morphology.builder import build_agreement_graph
    from src.morphology.graph import EDGE_TYPE_ORDER
    from src.evaluation.f05_scorer import evaluate_corpus

    t_cfg = cfg.get("training", {})
    batch_size = t_cfg.get("batch_size", 4)
    max_length = cfg.get("data", {}).get("max_seq_length", 64)
    epochs = t_cfg.get("max_epochs", 2)
    lr = t_cfg.get("learning_rate", 5e-5)
    grad_accum = t_cfg.get("gradient_accumulation_steps", 1)
    beam_width = cfg.get("evaluation", {}).get("beam_width", 2)

    device = torch.device("cpu")

    analyzer = MorphologicalAnalyzer(use_klpt=False)
    extractor = FeatureExtractor(analyzer=analyzer)

    model = MorphologyAwareGEC(
        model_name=cfg.get("model", {}).get("pretrained", "google/byt5-small"),
        max_length=max_length,
        agreement_loss_weight=t_cfg.get("agreement_loss_weight", 0.3),
    )
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info("MorphAware params: %s", f"{total_params:,}")

    num_types = len(EDGE_TYPE_ORDER)

    def build_morph_batch(sentences: list[str]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build morphological feature + agreement tensors for a batch."""
        batch_feats = []
        batch_agr = []
        batch_agr_labels = []

        for sent in sentences:
            # Word-level features
            word_feats = extractor.extract_features(sent)
            # Flatten to max_length (byte-level padding)
            feat_vec = [[0] * extractor.get_num_features()] * max_length
            for i, wf in enumerate(word_feats[:max_length]):
                feat_vec[i] = wf
            batch_feats.append(feat_vec)

            # Agreement graph
            graph = build_agreement_graph(sent, analyzer)
            stacked, _ = graph.to_typed_stacked_matrix()
            agr = torch.zeros(num_types, max_length, max_length, dtype=torch.int8)
            for t_idx in range(min(len(stacked), num_types)):
                n = min(len(stacked[t_idx]), max_length)
                for r in range(n):
                    m = min(len(stacked[t_idx][r]), max_length)
                    for c in range(m):
                        agr[t_idx, r, c] = stacked[t_idx][r][c]
            batch_agr.append(agr)

            # Agreement labels: per-position (simplified — class 0 = correct)
            labels = torch.zeros(max_length, dtype=torch.long)
            for edge in graph.edges:
                # Mark positions involved in agreement edges
                if edge.source_idx < max_length:
                    try:
                        cls_idx = EDGE_TYPE_ORDER.index(edge.agreement_type) + 1
                    except ValueError:
                        cls_idx = 0
                    labels[edge.source_idx] = cls_idx
            batch_agr_labels.append(labels)

        return (
            torch.tensor(batch_feats, dtype=torch.long),
            torch.stack(batch_agr),
            torch.stack(batch_agr_labels),
        )

    class _MorphDataset(Dataset):
        def __init__(self, srcs, tgts, tokenizer, ml):
            self.srcs, self.tgts = srcs, tgts
            self.tokenizer, self.ml = tokenizer, ml
        def __len__(self):
            return len(self.srcs)
        def __getitem__(self, idx):
            s = self.tokenizer(self.srcs[idx], max_length=self.ml,
                               truncation=True, padding="max_length", return_tensors="pt")
            t = self.tokenizer(self.tgts[idx], max_length=self.ml,
                               truncation=True, padding="max_length", return_tensors="pt")
            return {
                "input_ids": s["input_ids"].squeeze(0),
                "attention_mask": s["attention_mask"].squeeze(0),
                "labels": t["input_ids"].squeeze(0),
                "src_text": self.srcs[idx],
            }

    train_ds = _MorphDataset(train_src, train_tgt, model.tokenizer, max_length)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    dev_ds = _MorphDataset(dev_src, dev_tgt, model.tokenizer, max_length)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = math.ceil(len(train_loader) / grad_accum) * epochs
    warmup_steps = t_cfg.get("warmup_steps", 5)
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR
    warmup_sched = LinearLR(optimizer, start_factor=0.1, total_iters=max(1, warmup_steps))
    cosine_sched = CosineAnnealingWarmRestarts(optimizer, T_0=max(1, total_steps - warmup_steps))
    scheduler = SequentialLR(optimizer, [warmup_sched, cosine_sched], milestones=[warmup_steps])

    best_f05 = 0.0
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        model.anneal_agreement_loss(epoch, t_cfg.get("agreement_warmup_epochs", 1))
        total_loss = 0.0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            src_texts = batch.pop("src_text")
            batch = {k: v.to(device) for k, v in batch.items()}

            morph_feats, agr_mask, agr_labels = build_morph_batch(list(src_texts))
            morph_feats = morph_feats.to(device)
            agr_mask = agr_mask.to(device)
            agr_labels = agr_labels.to(device)

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                morph_features=morph_feats,
                agreement_mask=agr_mask,
                agreement_labels=agr_labels,
            )
            loss = outputs["loss"] / grad_accum
            loss.backward()

            if (batch_idx + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            total_loss += outputs["loss"].item()
            if (batch_idx + 1) % 5 == 0:
                logger.info(
                    "  [MorphAware] Epoch %d Batch %d — loss: %.4f (seq2seq: %.4f, agr: %.4f)",
                    epoch + 1, batch_idx + 1, outputs["loss"].item(),
                    outputs["seq2seq_loss"].item(), outputs["agreement_loss"].item(),
                )

        avg_loss = total_loss / max(len(train_loader), 1)
        logger.info("Epoch %d/%d — train loss: %.4f", epoch + 1, epochs, avg_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in dev_loader:
                src_texts = batch.pop("src_text")
                batch = {k: v.to(device) for k, v in batch.items()}
                morph_feats, agr_mask, agr_labels = build_morph_batch(list(src_texts))
                o = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    morph_features=morph_feats.to(device),
                    agreement_mask=agr_mask.to(device),
                    agreement_labels=agr_labels.to(device),
                )
                val_loss += o["loss"].item()
        val_loss /= max(len(dev_loader), 1)

        # F0.5 on dev (without morphology for speed)
        hypotheses = []
        for s in dev_src:
            hypotheses.append(model.correct(s, num_beams=beam_width))
        metrics = evaluate_corpus(dev_src, hypotheses, dev_tgt)
        logger.info("Epoch %d — val loss: %.4f | %s", epoch + 1, val_loss, metrics)

        if metrics.f05 >= best_f05:
            best_f05 = metrics.f05
            torch.save(model.state_dict(), output_dir / "best_model.pt")

    torch.save(model.state_dict(), output_dir / "final_model.pt")
    logger.info("MorphAware training done. Best F0.5: %.4f", best_f05)
    return best_f05


# ---------------------------------------------------------------------------
# Phase 6 — Evaluation on test set
# ---------------------------------------------------------------------------

def evaluate_test(
    test_src: list[str], test_tgt: list[str],
    baseline_dir: Path, morphaware_dir: Path,
    cfg: dict,
) -> dict:
    """Load best models and evaluate on test set."""
    logger.info("=== Phase 6: Test evaluation ===")
    import torch
    from src.model.baseline import BaselineGEC
    from src.model.morphology_aware import MorphologyAwareGEC
    from src.evaluation.f05_scorer import evaluate_corpus

    max_length = cfg.get("data", {}).get("max_seq_length", 64)
    beam_width = cfg.get("evaluation", {}).get("beam_width", 2)
    results = {}

    # Evaluate baseline
    baseline_ckpt = baseline_dir / "best_model.pt"
    if baseline_ckpt.exists():
        model = BaselineGEC(
            model_name=cfg.get("model", {}).get("pretrained", "google/byt5-small"),
            max_length=max_length,
        )
        model.load_state_dict(torch.load(baseline_ckpt, map_location="cpu", weights_only=True))
        model.eval()
        hyps = model.correct_batch(test_src, num_beams=beam_width)
        metrics = evaluate_corpus(test_src, hyps, test_tgt)
        results["baseline"] = metrics
        logger.info("Baseline test: %s", metrics)

        # Print sample corrections
        logger.info("--- Baseline sample corrections ---")
        for i in range(min(3, len(test_src))):
            logger.info("  SRC: %s", test_src[i][:80])
            logger.info("  HYP: %s", hyps[i][:80])
            logger.info("  REF: %s", test_tgt[i][:80])
            logger.info("  ---")

    # Evaluate morphaware
    morphaware_ckpt = morphaware_dir / "best_model.pt"
    if morphaware_ckpt.exists():
        model = MorphologyAwareGEC(
            model_name=cfg.get("model", {}).get("pretrained", "google/byt5-small"),
            max_length=max_length,
        )
        model.load_state_dict(torch.load(morphaware_ckpt, map_location="cpu", weights_only=True))
        model.eval()
        hyps = [model.correct(s, num_beams=beam_width) for s in test_src]
        metrics = evaluate_corpus(test_src, hyps, test_tgt)
        results["morphaware"] = metrics
        logger.info("MorphAware test: %s", metrics)

    return results


# ---------------------------------------------------------------------------
# Phase 7 — Agreement accuracy check
# ---------------------------------------------------------------------------

def check_agreement_accuracy(test_tgt: list[str]) -> None:
    """Run agreement accuracy checker on clean test sentences."""
    logger.info("=== Phase 7: Agreement accuracy check ===")
    from src.evaluation.agreement_accuracy import AgreementChecker
    from src.morphology.analyzer import MorphologicalAnalyzer

    analyzer = MorphologicalAnalyzer(use_klpt=False)
    checker = AgreementChecker(analyzer=analyzer)

    total_checks = 0
    total_passed = 0
    for sent in test_tgt[:10]:
        result = checker.check_sentence(sent)
        total_checks += result.checks_total
        total_passed += result.checks_passed
        if result.violations:
            logger.info("  Violations in: %s", sent[:60])
            for v in result.violations[:3]:
                logger.info("    %s", v)

    acc = total_passed / total_checks if total_checks > 0 else 0.0
    logger.info("Agreement accuracy on clean text: %.1f%% (%d/%d checks)",
                acc * 100, total_passed, total_checks)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Smoke test — full pipeline verification")
    parser.add_argument("--config", default="configs/laptop_smoke.yaml")
    parser.add_argument("--skip-morphaware", action="store_true",
                        help="Skip morphology-aware training (baseline only)")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
    else:
        logger.error("Config not found: %s", cfg_path)
        return

    data_dir = Path(cfg["data"]["splits_dir"])
    save_dir = Path(cfg["training"]["save_dir"])

    overall_start = time.perf_counter()
    results_summary = {}

    # Phase 1: Data
    with timed("Phase 1: Data verification"):
        train_src, train_tgt, dev_src, dev_tgt, test_src, test_tgt = verify_data(data_dir)

    # Phase 2: Error pipeline
    with timed("Phase 2: Error pipeline"):
        verify_error_pipeline(test_tgt)

    # Phase 3: Morphology
    with timed("Phase 3: Morphology"):
        verify_morphology(test_tgt)

    # Phase 4: Baseline training
    with timed("Phase 4: Baseline training"):
        baseline_f05 = train_baseline(
            train_src, train_tgt, dev_src, dev_tgt,
            cfg, save_dir / "baseline",
        )
        results_summary["baseline_best_f05"] = baseline_f05

    # Phase 5: MorphAware training
    if not args.skip_morphaware:
        with timed("Phase 5: MorphAware training"):
            morph_f05 = train_morphaware(
                train_src, train_tgt, dev_src, dev_tgt,
                cfg, save_dir / "morphaware",
            )
            results_summary["morphaware_best_f05"] = morph_f05

    # Phase 6: Test evaluation
    with timed("Phase 6: Test evaluation"):
        test_results = evaluate_test(
            test_src, test_tgt,
            save_dir / "baseline",
            save_dir / "morphaware",
            cfg,
        )
        for model_name, metrics in test_results.items():
            results_summary["%s_test_f05" % model_name] = metrics.f05
            results_summary["%s_test_precision" % model_name] = metrics.precision
            results_summary["%s_test_recall" % model_name] = metrics.recall

    # Phase 7: Agreement accuracy
    with timed("Phase 7: Agreement accuracy"):
        check_agreement_accuracy(test_tgt)

    total_time = time.perf_counter() - overall_start

    # Final summary
    logger.info("=" * 60)
    logger.info("SMOKE TEST COMPLETE — %.1f s total", total_time)
    logger.info("=" * 60)
    for k, v in results_summary.items():
        logger.info("  %s: %.4f", k, v)
    logger.info("=" * 60)

    # Save results
    results_file = save_dir / "smoke_results.json"
    with open(results_file, "w", encoding="utf-8") as fh:
        json.dump(results_summary, fh, indent=2)
    logger.info("Results saved to %s", results_file)

    logger.info("All pipeline components verified successfully.")


if __name__ == "__main__":
    main()
