"""
Inter-Rater Agreement Analysis

Reads human evaluation JSONL ratings saved by the web app and computes
Cohen's kappa and percentage agreement between rater pairs.
"""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def load_ratings(eval_dir: Path) -> dict[str, list[dict]]:
    """Load all rater JSONL files from the evaluation directory.

    Returns:
        Dict mapping rater_id to list of rating records.
    """
    ratings: dict[str, list[dict]] = {}
    for f in eval_dir.glob("ratings_*.jsonl"):
        rater_id = f.stem.replace("ratings_", "")
        entries = []
        with open(f, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        ratings[rater_id] = entries
        logger.info("Loaded %d ratings from %s", len(entries), f.name)
    return ratings


def _build_pairwise_labels(
    ratings_a: list[dict],
    ratings_b: list[dict],
) -> tuple[list[str], list[str]]:
    """Align ratings from two raters on the same (source, corrected) pairs.

    Returns two parallel lists of rating labels for overlapping items.
    """
    key_fn = lambda r: (r.get("source", ""), r.get("corrected", ""))
    map_b = {}
    for r in ratings_b:
        map_b[key_fn(r)] = r["rating"]

    labels_a, labels_b = [], []
    for r in ratings_a:
        k = key_fn(r)
        if k in map_b:
            labels_a.append(r["rating"])
            labels_b.append(map_b[k])
    return labels_a, labels_b


def percentage_agreement(labels_a: list[str], labels_b: list[str]) -> float:
    """Compute simple percentage agreement."""
    if not labels_a:
        return 0.0
    agree = sum(1 for a, b in zip(labels_a, labels_b) if a == b)
    return agree / len(labels_a)


def cohens_kappa(labels_a: list[str], labels_b: list[str]) -> float:
    """Compute Cohen's kappa for two raters.

    Handles arbitrary categorical labels.
    """
    n = len(labels_a)
    if n == 0:
        return 0.0

    categories = sorted(set(labels_a) | set(labels_b))
    cat_idx = {c: i for i, c in enumerate(categories)}
    k = len(categories)

    # Build confusion matrix
    matrix = [[0] * k for _ in range(k)]
    for a, b in zip(labels_a, labels_b):
        matrix[cat_idx[a]][cat_idx[b]] += 1

    p_o = sum(matrix[i][i] for i in range(k)) / n  # observed agreement

    # Expected agreement by chance
    p_e = 0.0
    for i in range(k):
        row_sum = sum(matrix[i])
        col_sum = sum(matrix[r][i] for r in range(k))
        p_e += (row_sum / n) * (col_sum / n)

    if p_e > 1.0 - 1e-10:
        return 0.0
    return (p_o - p_e) / (1.0 - p_e)


def compute_inter_rater_agreement(
    eval_dir: Path,
) -> dict[str, dict]:
    """Compute pairwise inter-rater agreement for all rater pairs.

    Returns:
        Dict mapping rater pair string to metrics dict with
        'kappa', 'agreement', and 'n_overlap'.
    """
    all_ratings = load_ratings(eval_dir)
    rater_ids = sorted(all_ratings.keys())
    results = {}

    for i in range(len(rater_ids)):
        for j in range(i + 1, len(rater_ids)):
            ra, rb = rater_ids[i], rater_ids[j]
            labels_a, labels_b = _build_pairwise_labels(
                all_ratings[ra], all_ratings[rb]
            )
            if not labels_a:
                continue
            kappa = cohens_kappa(labels_a, labels_b)
            pct = percentage_agreement(labels_a, labels_b)
            pair_key = f"{ra}||vs||{rb}"
            results[pair_key] = {
                "kappa": round(kappa, 4),
                "agreement": round(pct, 4),
                "n_overlap": len(labels_a),
            }
            logger.info(
                "%s vs %s: kappa=%.4f agreement=%.1f%% (n=%d)",
                ra, rb, kappa, pct * 100, len(labels_a),
            )
    return results
