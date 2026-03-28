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
    dropped = 0
    total_lines = 0
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            total_lines = line_num
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning("Line %d invalid JSON: %s — skipped", line_num, e)
                dropped += 1
                continue
            if "source" not in record or "target" not in record:
                logger.warning("Line %d missing source/target — skipped", line_num)
                dropped += 1
                continue
            pairs.append(record)
    logger.info("Loaded %d pairs from %s (%d lines read)", len(pairs), path, total_lines)
    if dropped > 0:
        pct = 100.0 * dropped / max(total_lines, 1)
        log_fn = logger.warning if pct > 1.0 else logger.info
        log_fn("Dropped %d lines (%.1f%%) from %s", dropped, pct, path)
    return pairs


def split_pairs(
    pairs: list[dict],
    train_ratio: float = 0.8,
    dev_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    stratify_key: Optional[str] = None,
    group_key: Optional[str] = None,
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
        group_key: If set, group pairs by this key (e.g. 'source_id') and
                   split at the group level to prevent data leakage between
                   sentences from the same source article.

    Returns:
        (train, dev, test) tuple of lists.
    """
    assert abs(train_ratio + dev_ratio + test_ratio - 1.0) < 1e-6, \
        f"Ratios must sum to 1.0, got {train_ratio + dev_ratio + test_ratio}"

    rng = random.Random(seed)

    if group_key:
        return _group_split(pairs, train_ratio, dev_ratio, group_key, rng)

    if stratify_key:
        return _stratified_split(pairs, train_ratio, dev_ratio, stratify_key, rng)

    shuffled = list(pairs)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_dev = int(n * dev_ratio)
    # Ensure non-empty dev/test when dataset is small
    if n >= 3:
        n_dev = max(1, n_dev)
        n_train = max(1, n_train)
        # Avoid train consuming everything
        if n_train + n_dev >= n:
            n_train = n - n_dev - 1

    train = shuffled[:n_train]
    dev = shuffled[n_train:n_train + n_dev]
    test = shuffled[n_train + n_dev:]

    logger.info("Split: train=%d, dev=%d, test=%d", len(train), len(dev), len(test))

    # Check for data leakage: target sentences shared across splits
    leaked = check_leakage(train, dev, test)
    if leaked:
        logger.warning(
            "Data leakage detected: %d target sentences overlap between splits. "
            "Deduplicating dev/test by removing leaked items.", leaked,
        )
        train_targets = {p["target"] for p in train}
        dev = [p for p in dev if p["target"] not in train_targets]
        test_targets = train_targets | {p["target"] for p in dev}
        test = [p for p in test if p["target"] not in test_targets]
        logger.info("After dedup: train=%d, dev=%d, test=%d", len(train), len(dev), len(test))

    return train, dev, test


def check_leakage(
    train: list[dict], dev: list[dict], test: list[dict],
) -> int:
    """Count target sentences that appear in more than one split."""
    train_tgt = {p["target"] for p in train}
    dev_tgt = {p["target"] for p in dev}
    test_tgt = {p["target"] for p in test}
    return len(train_tgt & dev_tgt) + len(train_tgt & test_tgt) + len(dev_tgt & test_tgt)


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

    if len(buckets) == 1 and "unknown" in buckets:
        logger.warning(
            "Stratification by '%s' found no valid labels — "
            "all %d pairs mapped to 'unknown'. Falling back to random split.",
            key, len(pairs),
        )

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


def _group_split(
    pairs: list[dict],
    train_ratio: float,
    dev_ratio: float,
    group_key: str,
    rng: random.Random,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split at the group level to prevent data leakage.

    All sentences sharing the same ``group_key`` value stay in the same
    split.  Groups are shuffled and allocated greedily to train/dev/test
    until the target ratios are approximately met.
    """
    groups: dict[str, list[dict]] = {}
    for p in pairs:
        gid = str(p.get(group_key, "ungrouped"))
        groups.setdefault(gid, []).append(p)

    group_ids = list(groups.keys())
    rng.shuffle(group_ids)

    n = len(pairs)
    target_train = int(n * train_ratio)
    target_dev = int(n * dev_ratio)

    train, dev, test = [], [], []
    train_count, dev_count = 0, 0

    for gid in group_ids:
        sentences = groups[gid]
        if train_count < target_train:
            train.extend(sentences)
            train_count += len(sentences)
        elif dev_count < target_dev:
            dev.extend(sentences)
            dev_count += len(sentences)
        else:
            test.extend(sentences)

    rng.shuffle(train)
    rng.shuffle(dev)
    rng.shuffle(test)

    logger.info(
        "Group split by '%s': %d groups → train=%d, dev=%d, test=%d",
        group_key, len(group_ids), len(train), len(dev), len(test),
    )
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
    group_key: Optional[str] = None,
) -> dict[str, int]:
    """End-to-end: load pairs, split, save to output_dir.

    Writes train.jsonl, dev.jsonl, test.jsonl to output_dir.
    Returns a dict with split sizes.
    """
    pairs = load_pairs(input_path)
    train, dev, test = split_pairs(
        pairs, train_ratio, dev_ratio, test_ratio, seed, stratify_key,
        group_key=group_key,
    )

    save_split(train, output_dir / "train.jsonl")
    save_split(dev, output_dir / "dev.jsonl")
    save_split(test, output_dir / "test.jsonl")

    return {"train": len(train), "dev": len(dev), "test": len(test)}


def kfold_split(
    pairs: list[dict],
    k: int = 5,
    seed: int = 42,
    stratify_key: Optional[str] = None,
) -> list[tuple[list[dict], list[dict]]]:
    """K-fold cross-validation split.

    Args:
        pairs: List of sentence-pair dicts.
        k: Number of folds (default 5).
        seed: Random seed for reproducibility.
        stratify_key: If set, stratify by this dict key.

    Returns:
        List of k (train, val) tuples.
    """
    rng = random.Random(seed)

    if stratify_key:
        buckets: dict[str, list[dict]] = {}
        for p in pairs:
            label = p.get(stratify_key, "unknown")
            buckets.setdefault(label, []).append(p)
        # Shuffle each bucket then interleave for stratification
        indices: list[int] = []
        index_map = {i: p for i, p in enumerate(pairs)}
        ordered: list[dict] = []
        for label, items in buckets.items():
            rng.shuffle(items)
            ordered.extend(items)
        pairs = ordered
    else:
        pairs = list(pairs)
        rng.shuffle(pairs)

    fold_size = len(pairs) // k
    folds: list[tuple[list[dict], list[dict]]] = []
    for i in range(k):
        start = i * fold_size
        end = start + fold_size if i < k - 1 else len(pairs)
        val = pairs[start:end]
        train = pairs[:start] + pairs[end:]
        folds.append((train, val))

    logger.info("K-fold split: k=%d, fold_size~%d, total=%d", k, fold_size, len(pairs))
    return folds


def run_kfold_split(
    input_path: Path,
    output_dir: Path,
    k: int = 5,
    seed: int = 42,
    stratify_key: Optional[str] = None,
) -> list[dict[str, int]]:
    """End-to-end: load pairs, k-fold split, save to output_dir.

    Writes fold_0/train.jsonl, fold_0/val.jsonl, ..., fold_{k-1}/ to output_dir.
    Returns a list of dicts with fold sizes.
    """
    pairs = load_pairs(input_path)
    folds = kfold_split(pairs, k=k, seed=seed, stratify_key=stratify_key)
    sizes = []
    for i, (train, val) in enumerate(folds):
        fold_dir = output_dir / f"fold_{i}"
        save_split(train, fold_dir / "train.jsonl")
        save_split(val, fold_dir / "val.jsonl")
        sizes.append({"fold": i, "train": len(train), "val": len(val)})
    return sizes
