"""Step 15: strip category/metadata contamination from the book corpus.

balanced_corpus.txt and clean_corpus.txt are stored as TSV:

    <category>\t<sentence>
    <category>\t<n>\t<arabic>\t<kurdish>

The pool generator in step 14 read each *whole line* as a sentence, so the
leading ``linguistics\t`` / ``islamic_studies\t`` prefix (and stray OCR Latin
glosses like "Page", "Wants Social") leaked into every training target. The
model then learned to emit a domain label at the start of every correction,
which is what tanked precision in the phase-2 run (FP floor ~3340, constant).

This script removes the category column, drops residual standalone Latin/OCR
tokens, collapses whitespace, and keeps only lines with real Sorani content.
splits_v2's dev/test are untouched (they were already clean).

Usage
-----
    python scripts/15_clean_corpus.py \
        --input data/balanced/balanced_corpus.txt \
        --output data/balanced/balanced_corpus_clean.txt
"""

import argparse
import re
import unicodedata

CATEGORY_SET = {
    "economics", "history", "islamic_studies", "law",
    "linguistics", "sciences", "social_sciences",
}

# Standalone ASCII-Latin token (OCR glosses, page markers) — removed wholesale.
LATIN_TOKEN = re.compile(r"^[A-Za-z][A-Za-z_]*$")
# Bare "Page 12" style markers.
PAGE_MARKER = re.compile(r"^[Pp]age\b.*")
ARABIC_SCRIPT = re.compile(r"[\u0600-\u06FF\u0750-\u077F]")


def clean_line(line: str) -> str | None:
    parts = line.rstrip("\n").split("\t")
    # 1) drop a leading category column.
    if parts and parts[0].strip() in CATEGORY_SET:
        parts = parts[1:]
    # 2) join remaining columns, then tokenise on whitespace.
    text = " ".join(p for p in parts if p.strip())
    text = unicodedata.normalize("NFC", text)
    toks = text.split()
    kept = []
    for tok in toks:
        if PAGE_MARKER.match(tok):
            continue
        # strip standalone Latin OCR tokens, but keep tokens that contain any
        # Arabic-script character (real Sorani content mixed with digits etc.)
        if LATIN_TOKEN.match(tok) and not ARABIC_SCRIPT.search(tok):
            continue
        kept.append(tok)
    out = " ".join(kept).strip()
    # 3) require some real Sorani content.
    if len(out) < 3 or not ARABIC_SCRIPT.search(out):
        return None
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/balanced/balanced_corpus.txt")
    ap.add_argument("--output", default="data/balanced/balanced_corpus_clean.txt")
    args = ap.parse_args()

    n_in = n_out = n_drop = 0
    seen = set()
    with open(args.input, encoding="utf-8") as fin, \
         open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            n_in += 1
            cleaned = clean_line(line)
            if cleaned is None:
                n_drop += 1
                continue
            if cleaned in seen:
                n_drop += 1
                continue
            seen.add(cleaned)
            fout.write(cleaned + "\n")
            n_out += 1

    print(f"input lines:   {n_in}")
    print(f"written lines: {n_out}")
    print(f"dropped:       {n_drop} (empty/no-Sorani/dup)")
    print(f"output:        {args.output}")


if __name__ == "__main__":
    main()
