import hashlib
import logging
import os
import urllib.request
import urllib.error
from pathlib import Path

logger = logging.getLogger(__name__)

# SHA-256 of the known-good ckb-Arab.dic file.  Update this value
# when intentionally upgrading to a new lexicon version.
_EXPECTED_SHA256 = None  # Set after first verified download


def _sha256(path: Path) -> str:
    """Compute hex SHA-256 digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def download_lexicon():
    base_url = "https://raw.githubusercontent.com/sinaahmadi/KurdishHunspell/master/ckb/ckb-Arab.dic"
    dest_dir = Path(__file__).resolve().parent.parent / "data" / "lexicon"
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    dic_path = dest_dir / "ckb-Arab.dic"
    
    logger.info("Downloading Sina Ahmadi's Kurdish lexicon data to %s...", dic_path)
    req = urllib.request.Request(base_url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            content = response.read()
            with open(dic_path, "wb") as f:
                f.write(content)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
        logger.error("Error downloading lexicon: %s", e)
        raise SystemExit(1) from e
            
    logger.info("Downloaded %d bytes.", len(content))

    # PIPE-13: Verify checksum for reproducibility
    actual_hash = _sha256(dic_path)
    logger.info("SHA-256: %s", actual_hash)
    if _EXPECTED_SHA256 is not None and actual_hash != _EXPECTED_SHA256:
        logger.warning(
            "Checksum mismatch! Expected: %s  Got: %s  "
            "The upstream lexicon may have changed. Verify manually.",
            _EXPECTED_SHA256, actual_hash,
        )
    elif _EXPECTED_SHA256 is None:
        logger.info(
            "No expected checksum configured. To lock this version, set "
            "_EXPECTED_SHA256 = \"%s\" in this script.", actual_hash,
        )

    logger.info("Lexicon download complete.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    download_lexicon()
