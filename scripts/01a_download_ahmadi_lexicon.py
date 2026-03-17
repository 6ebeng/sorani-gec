import os
import urllib.request
from pathlib import Path

def download_lexicon():
    base_url = "https://raw.githubusercontent.com/sinaahmadi/KurdishHunspell/master/ckb/ckb-Arab.dic"
    dest_dir = Path(__file__).resolve().parent.parent / "data" / "lexicon"
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    dic_path = dest_dir / "ckb-Arab.dic"
    
    print(f"Downloading Sina Ahmadi's Kurdish lexicon data to {dic_path}...")
    req = urllib.request.Request(base_url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response:
        content = response.read()
        with open(dic_path, "wb") as f:
            f.write(content)
            
    print(f"Downloaded {len(content)} bytes.")
    print("Lexicon download complete. We can now use Sina Ahmadi's data purely in Python.")

if __name__ == "__main__":
    download_lexicon()
