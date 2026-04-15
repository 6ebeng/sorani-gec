"""
Step 1-KTC: Re-collect raw data with KTC category labels.

Combines:
  - Pre-collected OCR books (from backup) → labelled 'linguistics'
  - KTC corpus (cloned in data/ktc/) → per-category via KTC_CATEGORY_MAP

Writes per-category .txt files into data/raw/ for downstream balancing.
"""

import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.corpus_catalog import KTC_CATEGORY_MAP

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
KTC_DIR = Path("data/ktc")
OCR_BACKUP = RAW_DIR / "_academic_backup.txt"


def collect_ocr_as_linguistics() -> int:
    """Read the pre-collected OCR backup and write as linguistics.txt."""
    if not OCR_BACKUP.exists():
        logger.error("OCR backup not found: %s", OCR_BACKUP)
        return 0

    out = RAW_DIR / "linguistics.txt"
    lines = OCR_BACKUP.read_text("utf-8", errors="replace").splitlines()
    # Strip any accidental tab-prefixed lines (morphological tables)
    clean = []
    for line in lines:
        # If line has tab, it's likely a table entry — take only the text part
        if "\t" in line:
            text = line.split("\t", 1)[1].strip()
        else:
            text = line.strip()
        if text and len(text) > 10:
            clean.append(text)

    out.write_text("\n".join(clean) + "\n", encoding="utf-8")
    logger.info("linguistics (OCR): %d sentences → %s", len(clean), out)
    return len(clean)


def collect_ktc() -> dict[str, int]:
    """Read KTC repo and write per-category files."""
    if not KTC_DIR.exists():
        logger.error("KTC dir not found: %s", KTC_DIR)
        return {}

    counts: dict[str, int] = {}
    for ktc_cat, our_cat in sorted(KTC_CATEGORY_MAP.items()):
        cat_path = KTC_DIR / ktc_cat
        if not cat_path.is_dir():
            logger.warning("Missing KTC directory: %s", cat_path)
            continue

        out = RAW_DIR / f"{our_cat}.txt"
        sentences = []
        for txt_file in sorted(cat_path.rglob("*.txt")):
            text = txt_file.read_text("utf-8", errors="replace")
            for line in text.splitlines():
                line = line.strip()
                if line and len(line) > 10:
                    sentences.append(line)

        if sentences:
            # Append if file already exists (multiple KTC dirs → same category)
            with open(out, "a", encoding="utf-8") as f:
                for s in sentences:
                    f.write(s + "\n")
            prev = counts.get(our_cat, 0)
            counts[our_cat] = prev + len(sentences)
            logger.info(
                "  %s → %s: %d sentences (total: %d)",
                ktc_cat, our_cat, len(sentences), counts[our_cat],
            )

    return counts


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Remove old per-category files (not backups)
    for f in RAW_DIR.glob("*.txt"):
        if not f.name.startswith("_") and not f.name.startswith("."):
            f.unlink()

    logger.info("=== Collecting OCR books as 'linguistics' ===")
    ocr_count = collect_ocr_as_linguistics()

    logger.info("=== Collecting KTC corpus ===")
    ktc_counts = collect_ktc()

    logger.info("=== Summary ===")
    total = ocr_count
    logger.info("  linguistics (OCR):  %6d", ocr_count)
    for cat, n in sorted(ktc_counts.items(), key=lambda x: -x[1]):
        logger.info("  %-20s %6d", cat, n)
        total += n
    logger.info("  TOTAL:              %6d", total)
    logger.info("Category files written to %s", RAW_DIR)


if __name__ == "__main__":
    main()
