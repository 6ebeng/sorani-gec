"""
Data Integrity: Compute SHA-256 hashes for corpus reproducibility

After running the data pipeline (scripts 01–04), this script records
SHA-256 hashes of all data files so that corpus integrity can be
verified later. The hash manifest is saved as a JSON file in the
data directory.

Usage:
    python scripts/11_hash_data.py
    python scripts/11_hash_data.py --data-dir data --output data/sha256_manifest.json
"""

import argparse
import hashlib
import json
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest for a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Compute SHA-256 hashes for data files")
    parser.add_argument("--data-dir", default="data",
                        help="Root data directory to scan")
    parser.add_argument("--output", default="data/sha256_manifest.json",
                        help="Output manifest file")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error("Data directory not found: %s", data_dir)
        return

    manifest: dict[str, dict] = {}
    file_count = 0

    for path in sorted(data_dir.rglob("*")):
        if path.is_file() and path.name != ".gitkeep":
            rel = str(path.relative_to(data_dir)).replace("\\", "/")
            digest = sha256_file(path)
            size_bytes = path.stat().st_size
            manifest[rel] = {
                "sha256": digest,
                "size_bytes": size_bytes,
            }
            file_count += 1
            logger.info("  %s  %s (%d bytes)", digest[:16], rel, size_bytes)

    if file_count == 0:
        logger.warning("No data files found in %s — run the data pipeline first.", data_dir)
        return

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Manifest written to %s (%d files)", output_path, file_count)


if __name__ == "__main__":
    main()
