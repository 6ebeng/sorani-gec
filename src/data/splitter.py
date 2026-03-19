"""
Train / Dev / Test Splitter for Sorani Kurdish GEC

Splits a corpus of (source, target) sentence pairs into training,
development, and test sets according to configurable ratios.
Supports stratified splitting by error type when annotations are present.
"""

import json
import logging
import random
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def load_pairs(path: Path) -> list[dict]:
    """Load sentence pairs from a JSONL file.

    Each line is a JSON object with at least 'source' and 'target' keys.
    Optional: 'error_type', 'original_clean', 'metadata'.
    """
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning("Line %d invalid JSON: %s — skipped", line_num, e)
                continue
            if "source" not in record or "target" not in record:
                logger.warning("Line %d missing source/target — skipped", line_num)
                continue
            pairs.append(record)
    logger.info("Loaded %d pairs from %s", len(pairs), path)
    return pairs


def split_pairs(
    pairs: list[dict],
    train_ratio: float = 0.8,
    dev_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    stratify_key: Optional[str] = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split pairs into train / dev / test sets.

    Args:
        pairs: List of sentence-pair dicts.
        train_ratio: Fraction for training (default 0.8).
        dev_ratio: Fraction for development (default 0.1).
        test_ratio: Fraction for test (default 0.1).
        seed: Random seed for reproducibility.
        stratify_key: If set, stratify by this dict key (e.g. 'error_type')
                      so each split has proportional representation.

    Returns:
        (train, dev, test) tuple of lists.
    """
    assert abs(train_ratio + dev_ratio + test_ratio - 1.0) < 1e-6, \
        f"Ratios must sum to 1.0, got {train_ratio + dev_ratio + test_ratio}"

    rng = random.Random(seed)

    if stratify_key:
        return _stratified_split(pairs, train_ratio, dev_ratio, stratify_key, rng)

    shuffled = list(pairs)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_dev = int(n * dev_ratio)

    train = shuffled[:n_train]
    dev = shuffled[n_train:n_train + n_dev]
    test = shuffled[n_train + n_dev:]

    logger.info("Split: train=%d, dev=%d, test=%d", len(train), len(dev), len(test))
    return train, dev, test


def _stratified_split(
    pairs: list[dict],
    train_ratio: float,
    dev_ratio: float,
    key: str,
    rng: random.Random,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Stratified split preserving distribution of `key` across splits."""
    buckets: dict[str, list[dict]] = {}
    for p in pairs:
        label = p.get(key, "unknown")
        buckets.setdefault(label, []).append(p)

    train, dev, test = [], [], []
    for label, items in buckets.items():
        rng.shuffle(items)
        n = len(items)
        n_train = int(n * train_ratio)
        n_dev = int(n * dev_ratio)
        train.extend(items[:n_train])
        dev.extend(items[n_train:n_train + n_dev])
        test.extend(items[n_train + n_dev:])

    rng.shuffle(train)
    rng.shuffle(dev)
    rng.shuffle(test)

    logger.info("Stratified split by '%s': train=%d, dev=%d, test=%d",
                key, len(train), len(dev), len(test))
    return train, dev, test


def save_split(pairs: list[dict], path: Path) -> None:
    """Save a split to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in pairs:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("Saved %d pairs to %s", len(pairs), path)


def run_split(
    input_path: Path,
    output_dir: Path,
    train_ratio: float = 0.8,
    dev_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    stratify_key: Optional[str] = None,
) -> dict[str, int]:
    """End-to-end: load pairs, split, save to output_dir.

    Writes train.jsonl, dev.jsonl, test.jsonl to output_dir.
    Returns a dict with split sizes.
    """
    pairs = load_pairs(input_path)
    train, dev, test = split_pairs(
        pairs, train_ratio, dev_ratio, test_ratio, seed, stratify_key
    )

    save_split(train, output_dir / "train.jsonl")
    save_split(dev, output_dir / "dev.jsonl")
    save_split(test, output_dir / "test.jsonl")

    return {"train": len(train), "dev": len(dev), "test": len(test)}
