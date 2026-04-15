"""
Step 2: Normalize and Clean Collected Text

Includes quality-gate filtering (PIPE-11):
  - Drops sentences with >30% non-Arabic-script characters
  - Drops sentences shorter than 5 tokens or longer than 150 tokens
  - Drops sentences matching known Wikipedia boilerplate patterns

Usage:
    python scripts/02_normalize.py [--input data/raw] [--output data/clean]
"""

import argparse
import logging
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.normalizer import SoraniNormalizer, deduplicate_sentences
from src.data.sanitizer import (
    SoraniSanitizer, _LEADING_MARKER_RE, _SUPERSCRIPT_RE,
    _LATIN_WORD_RE, _count_tokens_zwnj,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# PIPE-11: Quality-gate patterns
_ARABIC_SCRIPT_RE = re.compile(r'[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]')
_BOILERPLATE_PATTERNS = [
    "ئەم بابەتە بچووکە",       # Wikipedia stub notice
    "سەرچاوەکان دیاری نەکراون",  # "sources not specified"
    "ئەم بابەتە پێویستی بە",     # "this article needs"
    "بابەتی سەرەکی",             # "main article"
]
_MIN_TOKENS = 5
_MAX_TOKENS = 40
_MAX_NON_ARABIC_RATIO = 0.30
# ByT5 max_seq_length is 256 bytes; Sorani chars ~2 bytes each.
# Allow a small margin for special tokens/prefix.
_MAX_BYTES = 480

# Sentence must end with terminal punctuation (catches OCR fragments)
_TERMINAL_PUNCT_RE = re.compile(r'[.؟!۔\)]\s*$')
# Arrows / metalinguistic notation (←, →, space+space, space=space)
_FORMULA_RE = re.compile(r'[←→≠≈]|\s[+=]\s')
# Lines still starting with a list marker after stripping (digits, letters, etc.)
_RESIDUAL_MARKER_RE = re.compile(
    r'^\s*(?:'
    r'[\u06F0-\u06F90-9]+\s*[-\).:]'             # digit markers "۲-"
    r'|[\u06F0-\u06F90-9]{1,3}\s+(?=[\u0600-\u06FF])'  # bare number "۱ text"
    r'|[\u0620-\u06D5][\u0640\u200C]?\s*[-\).:]' # Arabic/Kurdish letter markers
    r'|[a-zA-Z]\s*[-\).:]'                        # Latin letter markers
    r'|[-\u2013\u2014]\s+'                         # standalone dash "- "
    r'|\u067e[\u06F0-\u06F90-9]+\s*[/:]'          # textbook markers "پ۵/"
    r')'
)

# Arabic tashkeel (vowel diacritics): fatḥa, ḍamma, kasra, shadda, sukūn, tanwīn
# Sorani Kurdish does not use these — their presence signals Classical Arabic /
# Quranic text that is not natural Sorani prose.
_ARABIC_TASHKEEL_RE = re.compile(r'[\u064B-\u0652]')
_MIN_TASHKEEL_COUNT = 3  # threshold: 3+ diacritics → Arabic text

# Double parentheses used as quotation / citation notation: (( ... ))
_DOUBLE_PAREN_RE = re.compile(r'\(\(|\)\)')

# Symbol sequences at line start: <==> , <<, >>, == etc.
_SYMBOL_PREFIX_RE = re.compile(r'^\s*[<>=]{2,}')

# Slash-delimited lists: /text, text.../
_SLASH_DELIMITER_RE = re.compile(r'/[^/\s]{2,}[^/]*/')

# Nested reference parentheses: (text(text))
_NESTED_REF_RE = re.compile(r'\([^)]*\([^)]*\)\s*\)')


def passes_quality_gate(sentence: str) -> tuple[bool, str]:
    """Check whether a sentence passes quality filters.

    Returns:
        (passes, reason) — reason is empty string if passes is True.
    """
    n_tokens = _count_tokens_zwnj(sentence)
    if n_tokens < _MIN_TOKENS:
        return False, "too_few_tokens"
    if n_tokens > _MAX_TOKENS:
        return False, "too_many_tokens"

    # Byte-length check: sentences exceeding the model's byte budget
    # will be truncated during training, wasting compute.
    if len(sentence.encode("utf-8")) > _MAX_BYTES:
        return False, "too_many_bytes"

    # Fragment detection: must end with terminal punctuation
    stripped = sentence.rstrip()
    if stripped.endswith(':') or stripped.endswith('...') or stripped.endswith('…'):
        return False, "fragment"
    if not _TERMINAL_PUNCT_RE.search(stripped):
        return False, "fragment"

    # Formula / metalinguistic notation
    if _FORMULA_RE.search(sentence):
        return False, "formula_notation"

    # Latin words (2+ consecutive Latin letters): formula variables,
    # untranslated terms, OCR of English content
    if _LATIN_WORD_RE.search(sentence):
        return False, "latin_content"

    # Residual numbered list markers (belt-and-suspenders after stripping)
    if _RESIDUAL_MARKER_RE.match(sentence):
        return False, "list_marker"

    # Symbol-heavy prefix: <==> , == , << etc.
    if _SYMBOL_PREFIX_RE.match(sentence):
        return False, "symbol_prefix"

    # Arabic tashkeel: signals Classical Arabic / Quranic text
    if len(_ARABIC_TASHKEEL_RE.findall(sentence)) >= _MIN_TASHKEEL_COUNT:
        return False, "arabic_tashkeel"

    # Double parentheses notation: (( ... ))
    if _DOUBLE_PAREN_RE.search(sentence):
        return False, "bracket_notation"

    # Slash-delimited lists: /text, text/
    if _SLASH_DELIMITER_RE.search(sentence):
        return False, "slash_delimiter"

    # Nested reference parentheses: (text(text))
    if _NESTED_REF_RE.search(sentence):
        return False, "nested_reference"

    # Check Arabic-script ratio (excluding whitespace and punctuation)
    non_space = re.sub(r'\s', '', sentence)
    if non_space:
        arabic_count = len(_ARABIC_SCRIPT_RE.findall(non_space))
        ratio = 1.0 - (arabic_count / len(non_space))
        if ratio > _MAX_NON_ARABIC_RATIO:
            return False, "low_arabic_ratio"

    # Check boilerplate
    for pattern in _BOILERPLATE_PATTERNS:
        if pattern in sentence:
            return False, "boilerplate"

    return True, ""


def main():
    parser = argparse.ArgumentParser(description="Normalize Sorani Kurdish text")
    parser.add_argument("--input", default="data/raw")
    parser.add_argument("--output", default="data/clean")
    parser.add_argument("--remove-diacritics", action="store_true")
    parser.add_argument("--skip-quality-gate", action="store_true",
                        help="Skip quality-gate filtering (PIPE-11)")
    parser.add_argument("--skip-sanitizer", action="store_true",
                        help="Skip SoraniSanitizer (page markers, Gemini artifacts, LaTeX)")
    parser.add_argument("--preserve-categories", action="store_true", default=False,
                        help="Read/write tab-separated category\\tsentence format")
    args = parser.parse_args()
    
    normalizer = SoraniNormalizer(
        remove_diacritics=args.remove_diacritics,
    )
    sanitizer = SoraniSanitizer(near_dup_threshold=1.0) if not args.skip_sanitizer else None
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    all_sentences = []
    all_categories: list[str] = []
    dropped_short = 0
    sanitizer_drops = 0
    resplit_count = 0
    quality_drops: dict[str, int] = {}
    
    for txt_file in sorted(input_path.glob("*.txt")):
        if txt_file.name.startswith("."):
            logger.info("Skipping hidden file: %s", txt_file.name)
            continue
        logger.info("Processing %s...", txt_file.name)
        with open(txt_file, "r", encoding="utf-8-sig", errors="replace") as f:
            for line in f:
                raw = line.strip()
                cat = None
                if args.preserve_categories and "\t" in raw:
                    cat, raw = raw.split("\t", 1)
                # Sanitize before normalizing (strip page markers, OCR junk, LaTeX)
                if sanitizer is not None:
                    cleaned = sanitizer.sanitize_line(raw)
                    if cleaned is None:
                        sanitizer_drops += 1
                        continue
                    raw = cleaned
                normalized = normalizer.normalize(raw)
                if normalized and len(normalized) > 20:
                    # Re-split long sentences at clause boundaries
                    parts = SoraniSanitizer.split_long_sentence(
                        normalized, max_tokens=_MAX_TOKENS,
                    )
                    if len(parts) > 1:
                        resplit_count += len(parts) - 1
                    for part in parts:
                        # Strip list markers/superscripts that appear mid-line
                        part = _LEADING_MARKER_RE.sub('', part)
                        part = _SUPERSCRIPT_RE.sub('', part).strip()
                        if len(part) <= 20:
                            dropped_short += 1
                            continue
                        # PIPE-11: Quality-gate filtering
                        if not args.skip_quality_gate:
                            passes, reason = passes_quality_gate(part)
                            if not passes:
                                quality_drops[reason] = quality_drops.get(reason, 0) + 1
                                continue
                        all_sentences.append(part)
                        all_categories.append(cat if cat else "")
                elif normalized:
                    dropped_short += 1
    
    if dropped_short:
        logger.info("Dropped %d lines shorter than 20 chars", dropped_short)
    if sanitizer_drops:
        logger.info("Sanitizer dropped %d lines total", sanitizer_drops)
        if sanitizer is not None and sanitizer.stats:
            for reason, count in sorted(sanitizer.stats.items(), key=lambda x: -x[1]):
                logger.info("  Sanitizer — %s: %d", reason, count)
    if resplit_count:
        logger.info("Re-split long sentences produced %d extra lines", resplit_count)
    if quality_drops:
        for reason, count in quality_drops.items():
            logger.info("Quality gate — dropped %d lines (%s)", count, reason)
        logger.info("Quality gate total drops: %d", sum(quality_drops.values()))
    
    logger.info("Total sentences before dedup: %d", len(all_sentences))
    if args.preserve_categories:
        # Deduplicate while keeping category alignment
        seen: set[str] = set()
        deduped_sents: list[str] = []
        deduped_cats: list[str] = []
        for s, c in zip(all_sentences, all_categories):
            if s not in seen:
                seen.add(s)
                deduped_sents.append(s)
                deduped_cats.append(c)
        all_sentences = deduped_sents
        all_categories = deduped_cats
    else:
        all_sentences = deduplicate_sentences(all_sentences)
    logger.info("Total sentences after dedup: %d", len(all_sentences))
    
    # Save cleaned corpus
    output_file = output_path / "clean_corpus.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        if args.preserve_categories:
            for sent, cat in zip(all_sentences, all_categories):
                f.write("%s\t%s\n" % (cat, sent))
        else:
            for sentence in all_sentences:
                f.write(sentence + "\n")
    
    logger.info("Clean corpus saved to %s", output_file)


if __name__ == "__main__":
    main()
