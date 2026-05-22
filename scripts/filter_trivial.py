"""
Phase 3 — R1: Filter trivial (source == target) pairs from data splits.

Creates data/splits_filtered/ with only non-trivial pairs.
Retains original splits as-is for reference.

Usage:
    python scripts/filter_trivial.py [--input data/splits] [--output data/splits_filtered]
"""
import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def filter_split(input_path: Path, output_path: Path) -> dict:
    """Filter a .jsonl split to keep only source != target pairs."""
    records = []
    total = 0
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            total += 1
            if rec.get("source", "") != rec.get("target", ""):
                records.append(rec)

    trivial = total - len(records)
    logger.info(
        "%s: %d total, %d trivial (%.1f%%), %d non-trivial kept",
        input_path.name, total, trivial, 100*trivial/max(total,1), len(records)
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return {"total": total, "trivial": trivial, "kept": len(records)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/splits")
    parser.add_argument("--output", default="data/splits_filtered")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {}
    for split in ["train", "dev", "test"]:
        src = input_dir / f"{split}.jsonl"
        dst = output_dir / f"{split}.jsonl"
        if src.exists():
            stats[split] = filter_split(src, dst)
        else:
            logger.warning("Missing: %s", src)

    # Write summary
    summary_path = output_dir / "filter_stats.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    logger.info("Summary written to %s", summary_path)
    print("\n=== Filtered splits summary ===")
    for split, s in stats.items():
        print(f"  {split}: {s['kept']}/{s['total']} pairs kept "
              f"({s['trivial']} trivial = {100*s['trivial']/max(s['total'],1):.1f}% removed)")


if __name__ == "__main__":
    main()
