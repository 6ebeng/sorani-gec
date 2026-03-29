import hashlib
import os
import urllib.request
import urllib.error
from pathlib import Path

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
    
    print(f"Downloading Sina Ahmadi's Kurdish lexicon data to {dic_path}...")
    req = urllib.request.Request(base_url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            content = response.read()
            with open(dic_path, "wb") as f:
                f.write(content)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
        print(f"Error downloading lexicon: {e}")
        raise SystemExit(1) from e
            
    print(f"Downloaded {len(content)} bytes.")

    # PIPE-13: Verify checksum for reproducibility
    actual_hash = _sha256(dic_path)
    print(f"SHA-256: {actual_hash}")
    if _EXPECTED_SHA256 is not None and actual_hash != _EXPECTED_SHA256:
        print(
            f"WARNING: Checksum mismatch!\n"
            f"  Expected: {_EXPECTED_SHA256}\n"
            f"  Got:      {actual_hash}\n"
            f"The upstream lexicon may have changed. Verify manually."
        )
    elif _EXPECTED_SHA256 is None:
        print(
            "No expected checksum configured. To lock this version, set\n"
            f"  _EXPECTED_SHA256 = \"{actual_hash}\"\n"
            "in this script."
        )

    print("Lexicon download complete.")

if __name__ == "__main__":
    download_lexicon()
