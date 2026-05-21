"""Re-segment data/clean/clean_corpus.txt into clean single-sentence records.

Pipeline per input line (category-tab-body):
  1. Split body on hard sentence terminators (. ؟ !), keeping the
     terminator on the preceding sentence.
  2. For each candidate sentence:
       - strip leading list markers (e.g. ``2-``, ``ب –``, ``•``, ``پ۲-``)
       - remove Arabic comma `،` and semicolon `؛` in place (collapse
         resulting double spaces) — sentences stay whole, but the model
         never sees clause-separating commas
       - require the sentence to still end with a hard terminator
       - require at least MIN_WORDS tokens
  3. Emit `category\tsentence` for each survivor.

Backs up the input as ``clean_corpus.txt.pre_resegment.bak`` on first run.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "clean" / "clean_corpus.txt"

MIN_WORDS = 5
# Run-on guard: sentences over these caps are almost always the result of an
# author omitting sentence-final punctuation, so two or more clauses got fused.
MAX_WORDS = 40
MAX_CHARS = 280

_HARD_SPLIT = re.compile(r'(?<=[.؟!])\s+')

# Characters that appear in Arabic but not in standard Sorani orthography.
# Their presence usually marks Arabic-script bibliographic citations or
# foreign-language fragments that slipped into the corpus during OCR.
_NON_SORANI_CHARS = set('\u0623\u0625\u0629\u0649\u0624\u064A\u0640')
# Latin ASCII letters — Kurdish citation tails (e.g. "f.", "p.") that
# survived collection are dropped wholesale; Sorani sentences shouldn't
# contain Latin letters.
_LATIN_RX = re.compile(r'[A-Za-z]')
# Trailing single Arabic-script letter + period — a truncated abbreviation
# like "... د." or "... ل." with no real sentence content after it.
_ABBREV_TAIL_RX = re.compile(r'(?:^|\s)[\u0621-\u06FF]{1,2}\.\s*$')
# Bracket / quote pairs that must balance for the sentence to be intact.
_BRACKETS = [('(', ')'), ('[', ']'), ('{', '}'), ('«', '»')]
# Three+ identical chars in a row (e.g. "ەەە", "ووو") — OCR noise.
_REPEAT_CHAR_RX = re.compile(r'([\u0621-\u06FF])\1{2,}')
# Doubled "ە" (heh with yeh above) — almost never valid in Sorani; reliably OCR noise.
_DOUBLE_AE_RX = re.compile(r'\u06D5\u06D5')
# Symbols that don't belong in clean Sorani prose: ASCII/Arabic double-quote,
# mid-sentence colon, percent, slash, backslash, equals, angle brackets,
# ampersand, asterisk, plus. Their presence flags citations, quoted material,
# stats fragments, or non-prose content.
# Symbols that don't belong in clean Sorani prose: ASCII/Arabic double-quote,
# mid-sentence colon, percent, slash, backslash, equals, angle brackets,
# ampersand, asterisk, plus, parens/brackets/braces, any digit (ASCII +
# Arabic-Indic + Persian), and long dashes (en/em/figure/horizontal-bar).
# Their presence flags citations, quoted material, stats fragments, page
# ranges, or other non-prose content.
_NOISE_SYMBOLS_RX = re.compile(r'["“”«»:%/\\=<>&*+()\[\]{}0-9\u0660-\u0669\u06F0-\u06F9\u2012\u2013\u2014\u2015\-\u2010\u2011\u2212]')
# Same short token repeated three times with whitespace ("و و و").
_REPEAT_TOKEN_RX = re.compile(r'(?:(?<=\s)|^)(\S{1,3})(?:\s+\1){2,}(?:(?=\s)|$)')
# Trailing 4-digit year (Latin or Arabic-Indic) — bibliographic citation tail.
_YEAR_TAIL_RX = re.compile(r'[\d\u0660-\u0669\u06F0-\u06F9]{4}\s*[.؟!]\s*$')
# Colon followed by terminator with nothing in between ("دەڵێت: .").
_EMPTY_QUOTE_RX = re.compile(r':\s*[.؟!]\s*$')
# Ellipsis followed by enumeration filler ("... و… هتد.") — open-ended list.
_ETC_TAIL_RX = re.compile(r'[\u2026]\s*هتد\s*[.؟!]?\s*$')
# Sentence ending with a hanging connector or stand-alone conjunction —
# e.g. "... و و .", "... بەڵام ." — the clause after it was dropped.
_DANGLING_TAIL_RX = re.compile(r'\s(و|یان|بەڵام|چونکە|کە|لەگەڵ|بۆ)(?:\s+\1)*\s*[.؟!]\s*$')
# Whitespace immediately before the terminator ("... وینەی .") —
# a token was dropped during OCR/extraction; the sentence is truncated.
_TRUNCATED_TAIL_RX = re.compile(r'\s[.؟!]\s*$')
# Sentence opening with a coordinating/subordinating connector — the
# previous clause was cut during segmentation; this is a continuation
# fragment, not a standalone sentence.
_LEAD_CONNECTOR_RX = re.compile(
    r'^(?:و|وە|بەڵام|کە|چونکە|لەگەڵ|یان|بۆیە|جا|پاشان|هەروەها|هەروەک|بۆ)\s'
)

_LIST_PREFIXES = [
    re.compile(r'^\s*پ\s*[\u0660-\u0669\u06F0-\u06F9\d]+\s*[-\u2010\u2011\u2012\u2013\u2014.)]\s*'),
    re.compile(r'^\s*[\u0660-\u0669\u06F0-\u06F9\d]+\s*[-\u2010\u2011\u2012\u2013\u2014./)]\s*'),
    re.compile(r'^\s*[\u0627\u0628\u067E\u062A\u062C\u0686\u062D\u062E\u062F\u0695\u0631\u0632\u0698\u0633\u0634\u0639\u063A\u0641\u06A4\u0642\u06A9\u06AF\u0644\u06B5\u0645\u0646\u0648\u06C6\u06BE\u06D5\u06CC\u06CE]\s*[-\u2010\u2011\u2012\u2013\u2014)]\s+'),
    re.compile(r'^\s*[-\u2010\u2011\u2012\u2013\u2014\u2022\u25CF\u25E6*]\s+'),
]

_COMMA_RX = re.compile(r'\s*[،؛]+\s*')
_WS_RX = re.compile(r'\s+')


def strip_list_markers(text: str) -> str:
    prev = None
    while prev != text:
        prev = text
        for rx in _LIST_PREFIXES:
            text = rx.sub('', text, count=1)
    return text


def clean_sentence(s: str) -> str | None:
    s = strip_list_markers(s).strip()
    if not s:
        return None
    s = _COMMA_RX.sub(' ', s)
    s = _WS_RX.sub(' ', s).strip()
    if not s:
        return None
    if s[-1] not in '.؟!':
        return None
    if len(s.split()) < MIN_WORDS:
        return None
    if len(s.split()) > MAX_WORDS:
        return None
    if len(s) > MAX_CHARS:
        return None
    if _LATIN_RX.search(s):
        return None
    if any(c in _NON_SORANI_CHARS for c in s):
        return None
    if _ABBREV_TAIL_RX.search(s):
        return None
    for op, cl in _BRACKETS:
        if s.count(op) != s.count(cl):
            return None
    if s.count('"') % 2 != 0:
        return None
    if _REPEAT_CHAR_RX.search(s):
        return None
    if _DOUBLE_AE_RX.search(s):
        return None
    if _NOISE_SYMBOLS_RX.search(s):
        return None
    if _REPEAT_TOKEN_RX.search(s):
        return None
    if _YEAR_TAIL_RX.search(s):
        return None
    if _EMPTY_QUOTE_RX.search(s):
        return None
    if _ETC_TAIL_RX.search(s):
        return None
    if _DANGLING_TAIL_RX.search(s):
        return None
    if _TRUNCATED_TAIL_RX.search(s):
        return None
    if _LEAD_CONNECTOR_RX.match(s):
        return None
    return s


def segment(text: str) -> list[str]:
    out: list[str] = []
    for raw in _HARD_SPLIT.split(text.strip()):
        cleaned = clean_sentence(raw)
        if cleaned:
            out.append(cleaned)
    return out


def main() -> None:
    backup = CORPUS.with_suffix(CORPUS.suffix + ".pre_resegment.bak")
    if not backup.exists():
        shutil.copy2(CORPUS, backup)

    lines = CORPUS.read_text(encoding="utf-8").splitlines()
    in_count = len(lines)
    out_lines: list[str] = []
    split_count = 0
    dropped = 0

    for line in lines:
        if "\t" in line:
            cat, body = line.split("\t", 1)
        else:
            cat, body = "general", line
        body = body.strip()
        if not body:
            continue
        segments = segment(body)
        if not segments:
            dropped += 1
            continue
        if len(segments) > 1:
            split_count += 1
        for s in segments:
            out_lines.append(f"{cat}\t{s}")

    CORPUS.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"input lines       : {in_count}")
    print(f"records split (>1): {split_count}")
    print(f"records dropped   : {dropped}")
    print(f"output sentences  : {len(out_lines)}")


if __name__ == "__main__":
    main()
