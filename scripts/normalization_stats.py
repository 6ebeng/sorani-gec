"""Corpus statistics for the Heh and digit normalisation choices.

Reviewer item R18: the normaliser rewrites non-initial U+0647 (Arabic heh) to
U+06D5 (Kurdish ae) and folds every digit system onto Extended Arabic-Indic
(U+06F0-U+06F9). The reviewer asked whether those edits are justified or risk
destroying meaningful distinctions. This script measures how often each
character actually occurs in the raw corpus and how often the rules fire, so
the thesis can report editorial decisions backed by counts rather than
assertion.
"""

import os
import re
import sys
import unicodedata
from collections import Counter

RAW_DIR = "data/raw"

HEH = "\u0647"          # ه Arabic heh
AE = "\u06d5"           # ە Kurdish ae (word-final/standalone vowel)
WESTERN = set("0123456789")
ARABIC_INDIC = set(chr(c) for c in range(0x0660, 0x066A))
EXT_ARABIC_INDIC = set(chr(c) for c in range(0x06F0, 0x06FA))

# A character counts as Arabic-script context if it is a letter in the Arabic block.
_ARABIC_LETTER = re.compile(r"[\u0600-\u06FF\u0750-\u077F]")


def iter_raw_text():
    for root, _, files in os.walk(RAW_DIR):
        for fn in files:
            if fn.endswith(".txt"):
                path = os.path.join(root, fn)
                try:
                    with open(path, encoding="utf-8") as f:
                        yield f.read()
                except Exception as e:  # noqa: BLE001
                    print(f"  skip {path}: {e}", file=sys.stderr)


def main():
    total_chars = 0
    heh_total = 0
    heh_initial = 0      # word-initial /h/ — preserved by the rule
    heh_noninitial = 0   # rewritten to AE by the rule
    ae_total = 0
    digit_counts = Counter()

    for text in iter_raw_text():
        text = unicodedata.normalize("NFC", text)
        total_chars += len(text)
        for i, ch in enumerate(text):
            if ch == HEH:
                heh_total += 1
                prev = text[i - 1] if i > 0 else ""
                if _ARABIC_LETTER.match(prev):
                    heh_noninitial += 1
                else:
                    heh_initial += 1
            elif ch == AE:
                ae_total += 1
            elif ch in WESTERN:
                digit_counts["western"] += 1
            elif ch in ARABIC_INDIC:
                digit_counts["arabic_indic"] += 1
            elif ch in EXT_ARABIC_INDIC:
                digit_counts["ext_arabic_indic"] += 1

    print(f"Total characters scanned: {total_chars:,}")
    print()
    print("HEH (U+0647) handling:")
    print(f"  total U+0647        : {heh_total:,}")
    print(f"  word-initial (kept) : {heh_initial:,}  ({100*heh_initial/max(heh_total,1):.1f}%)")
    print(f"  non-initial (->ae)  : {heh_noninitial:,}  ({100*heh_noninitial/max(heh_total,1):.1f}%)")
    print(f"  native ae U+06D5    : {ae_total:,}")
    print()
    print("DIGIT systems present:")
    tot_digits = sum(digit_counts.values())
    for k in ("western", "arabic_indic", "ext_arabic_indic"):
        c = digit_counts[k]
        print(f"  {k:18s}: {c:,}  ({100*c/max(tot_digits,1):.1f}%)")
    print(f"  total digits        : {tot_digits:,}")


if __name__ == "__main__":
    main()
