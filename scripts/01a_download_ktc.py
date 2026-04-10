"""
Step 1a: Download the Kurdish Textbooks Corpus (KTC)

Clones or updates the KurdishBLARK/KTC repository into data/ktc/ and
reports per-category file counts.

Usage:
    python scripts/01a_download_ktc.py [--output data/ktc]

Citation:
    Abdulrahman, R., Hassani, H., & Ahmadi, S. (2019).
    Developing a Fine-grained Corpus for a Less-resourced Language:
    the case of Kurdish. WiNLP ACL 2019, Florence, Italy.
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.corpus_catalog import KTC_CATEGORY_MAP

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

KTC_REPO_URL = "https://github.com/KurdishBLARK/KTC.git"
KTC_BRANCH = "master"


def clone_or_pull(dest: Path) -> None:
    """Clone the KTC repo, or pull if it already exists."""
    if (dest / ".git").is_dir():
        logger.info("KTC repo already cloned at %s — pulling latest changes", dest)
        subprocess.run(
            ["git", "-C", str(dest), "pull", "--ff-only"],
            check=True,
            capture_output=True,
            text=True,
        )
    else:
        logger.info("Cloning KTC repo to %s", dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--branch", KTC_BRANCH, "--depth", "1",
             KTC_REPO_URL, str(dest)],
            check=True,
            capture_output=True,
            text=True,
        )


def report_stats(ktc_dir: Path) -> dict[str, int]:
    """Count .txt files per KTC category directory."""
    stats: dict[str, int] = {}
    for ktc_cat in sorted(KTC_CATEGORY_MAP.keys()):
        cat_dir = ktc_dir / ktc_cat
        if not cat_dir.is_dir():
            logger.warning("Expected KTC directory not found: %s", cat_dir)
            stats[ktc_cat] = 0
            continue
        txt_files = list(cat_dir.rglob("*.txt"))
        stats[ktc_cat] = len(txt_files)
        mapped = KTC_CATEGORY_MAP[ktc_cat]
        logger.info("  %-15s → %-20s  %d files", ktc_cat, mapped, len(txt_files))
    total = sum(stats.values())
    logger.info("Total KTC text files: %d across %d categories", total, len(stats))
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the Kurdish Textbooks Corpus (KTC)",
    )
    parser.add_argument(
        "--output", default="data/ktc",
        help="Destination directory for the cloned KTC repo (default: data/ktc)",
    )
    args = parser.parse_args()

    dest = Path(args.output)
    clone_or_pull(dest)
    report_stats(dest)
    logger.info("KTC download complete. Use --source ktc in 01_collect_data.py")


if __name__ == "__main__":
    main()
