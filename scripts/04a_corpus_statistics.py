"""
Step 4a: Corpus Statistics and Error Distribution Analysis

Reads the synthetic annotations produced by 03_generate_errors.py and
outputs error-type frequency tables, per-type sentence counts, and
distribution plots saved to results/figures/.

Usage:
    python scripts/04a_corpus_statistics.py --input data/synthetic/annotations.jsonl
    python scripts/04a_corpus_statistics.py --input data/synthetic/annotations.jsonl --output results/figures
"""

import argparse
import json
import logging
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_annotations(path: Path) -> list[dict]:
    """Load JSONL annotations file."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSON at line %d", line_num)
    return records


def compute_statistics(records: list[dict]) -> dict:
    """Compute error distribution statistics from annotation records."""
    error_type_counter: Counter = Counter()
    sentence_lengths: list[int] = []
    error_counts_per_sentence: list[int] = []
    multi_error_sentences = 0

    for rec in records:
        # Count error types
        errors = rec.get("errors", [])
        types_in_sentence = set()
        for err in errors:
            etype = err.get("error_type", err.get("type", "unknown"))
            error_type_counter[etype] += 1
            types_in_sentence.add(etype)

        # Fall back to top-level error_type if no errors list
        if not errors and "error_type" in rec:
            error_type_counter[rec["error_type"]] += 1
            types_in_sentence.add(rec["error_type"])

        error_counts_per_sentence.append(len(errors) if errors else (1 if "error_type" in rec else 0))
        if len(types_in_sentence) > 1:
            multi_error_sentences += 1

        # Sentence length (by words)
        source = rec.get("source", rec.get("original", ""))
        sentence_lengths.append(len(source.split()))

    total_errors = sum(error_type_counter.values())
    total_sentences = len(records)

    stats = {
        "total_sentences": total_sentences,
        "total_errors": total_errors,
        "multi_error_sentences": multi_error_sentences,
        "unique_error_types": len(error_type_counter),
        "avg_errors_per_sentence": total_errors / max(total_sentences, 1),
        "avg_sentence_length_words": sum(sentence_lengths) / max(len(sentence_lengths), 1),
        "min_sentence_length": min(sentence_lengths) if sentence_lengths else 0,
        "max_sentence_length": max(sentence_lengths) if sentence_lengths else 0,
        "error_type_counts": dict(error_type_counter.most_common()),
        "error_type_percentages": {
            k: v / total_errors * 100
            for k, v in error_type_counter.most_common()
        } if total_errors > 0 else {},
    }
    return stats


def generate_plots(stats: dict, output_dir: Path) -> None:
    """Generate distribution plots and save to output_dir."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        logger.warning("matplotlib/seaborn not available — skipping plot generation")
        return

    sns.set_style("whitegrid")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Error type distribution bar chart
    type_counts = stats["error_type_counts"]
    if type_counts:
        fig, ax = plt.subplots(figsize=(12, 6))
        types = list(type_counts.keys())
        counts = list(type_counts.values())
        bars = ax.barh(types, counts, color=sns.color_palette("husl", len(types)))
        ax.set_xlabel("Count")
        ax.set_title("Error Type Distribution")
        ax.invert_yaxis()
        for bar, count in zip(bars, counts):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                    str(count), va="center", fontsize=8)
        plt.tight_layout()
        plt.savefig(output_dir / "error_type_distribution.png", dpi=150)
        plt.savefig(output_dir / "error_type_distribution.pdf")
        plt.close()
        logger.info("Saved error_type_distribution.png/pdf")

    # Error type percentage pie chart
    if type_counts:
        fig, ax = plt.subplots(figsize=(10, 10))
        top_n = 10
        sorted_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)
        if len(sorted_types) > top_n:
            top = sorted_types[:top_n]
            other = sum(v for _, v in sorted_types[top_n:])
            labels = [k for k, _ in top] + ["other"]
            sizes = [v for _, v in top] + [other]
        else:
            labels = [k for k, _ in sorted_types]
            sizes = [v for _, v in sorted_types]
        ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
        ax.set_title("Error Type Proportions")
        plt.tight_layout()
        plt.savefig(output_dir / "error_type_proportions.png", dpi=150)
        plt.savefig(output_dir / "error_type_proportions.pdf")
        plt.close()
        logger.info("Saved error_type_proportions.png/pdf")


def print_summary_table(stats: dict) -> None:
    """Print a formatted summary table to stdout."""
    print("\n" + "=" * 60)
    print("CORPUS STATISTICS")
    print("=" * 60)
    print(f"  Total sentences:          {stats['total_sentences']:,}")
    print(f"  Total error instances:    {stats['total_errors']:,}")
    print(f"  Unique error types:       {stats['unique_error_types']}")
    print(f"  Multi-error sentences:    {stats['multi_error_sentences']:,}")
    print(f"  Avg errors/sentence:      {stats['avg_errors_per_sentence']:.2f}")
    print(f"  Avg sentence length:      {stats['avg_sentence_length_words']:.1f} words")
    print(f"  Sentence length range:    {stats['min_sentence_length']}–{stats['max_sentence_length']} words")
    print()
    print("ERROR TYPE DISTRIBUTION:")
    print("-" * 60)
    print(f"  {'Error Type':<35} {'Count':>8} {'%':>7}")
    print("-" * 60)
    for etype, count in stats["error_type_counts"].items():
        pct = stats["error_type_percentages"].get(etype, 0)
        print(f"  {etype:<35} {count:>8,} {pct:>6.1f}%")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Corpus statistics and error distribution")
    parser.add_argument("--input", default="data/synthetic/annotations.jsonl",
                        help="Path to annotations JSONL file")
    parser.add_argument("--output", default="results/figures",
                        help="Directory for output plots and statistics JSON")
    parser.add_argument("--no-plots", action="store_true", default=False,
                        help="Skip plot generation")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)

    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        logger.error("Run scripts/03_generate_errors.py first to create annotations.")
        return

    logger.info("Loading annotations from %s", input_path)
    records = load_annotations(input_path)
    logger.info("Loaded %d records", len(records))

    stats = compute_statistics(records)
    print_summary_table(stats)

    # Save statistics JSON
    output_dir.mkdir(parents=True, exist_ok=True)
    stats_file = output_dir / "corpus_statistics.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    logger.info("Statistics saved to %s", stats_file)

    # Generate plots
    if not args.no_plots:
        generate_plots(stats, output_dir)

    logger.info("Done.")


if __name__ == "__main__":
    main()
