"""Shared Sorani Kurdish tokenization utilities.

Provides consistent tokenization across the pipeline, replacing bare
str.split() calls that mishandle zero-width characters and Arabic-script
punctuation boundaries.

Two tokenizers are exposed:
  - sorani_tokenize()       — regex-based; separates punctuation from words
  - sorani_word_tokenize()  — whitespace-based; drop-in for str.split()

Both remove ZWJ (U+200D) before tokenizing.  The regex tokenizer also
splits conjunctive و that runs into the next word (e.g. "وئەو" → "و ئەو").
"""

import re

# ---------------------------------------------------------------------------
# Regex matching individual Sorani tokens.  Mirrors (and replaces) the
# _TOKENIZE_PATTERN formerly in morphology/analyzer.py so that
# tokenization stays consistent across the whole codebase.
# ---------------------------------------------------------------------------
_SORANI_TOKEN_RE = re.compile(
    r'[\u0621-\u063A\u0641-\u064A\u066E-\u06D3\u06D5-\u06EF'
    r'\u06FA-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF'
    r'\u200C'                        # ZWNJ (morpheme boundary in Kurdish)
    r']+'
    r'|[\u06F0-\u06F90-9]+'         # digit runs
    r'|[a-zA-Z]+'                    # Latin letter runs
    r'|[^\s]'                        # single non-whitespace char (punctuation)
)

# Conjunctive و: و followed by a known word-initial letter without a space.
_WAW_CONJ_RE = re.compile(
    r'(?:(?<=\s)|^)و(?=[ئابتجحخدذرزسشصضطظعغفقكلمنهکگڵیێەپڕژڤڶ])'
)

# Punctuation marks that should not be preceded by a space when joining.
_NO_SPACE_BEFORE = frozenset('،؛؟۔.!:;)]}»')


def sorani_tokenize(text: str) -> list[str]:
    """Full regex-based Sorani tokenization.

    Removes ZWJ, splits conjunctive و, and separates punctuation into
    individual tokens.  Suitable for morphological analysis and any
    context where precise token boundaries matter.
    """
    text = text.replace('\u200d', '')        # strip ZWJ
    text = _WAW_CONJ_RE.sub('و ', text)      # split conjunctive و
    return [t for t in _SORANI_TOKEN_RE.findall(text) if t.strip()]


def sorani_word_tokenize(text: str) -> list[str]:
    """Whitespace-based Sorani tokenization with ZWJ removal.

    Drop-in replacement for ``str.split()`` on Sorani text.  Removes
    invisible ZWJ (U+200D) before splitting so downstream code never
    encounters this zero-width character inside tokens.  Token boundaries
    stay compatible with ``re.finditer(r'\\S+')`` on the cleaned text.
    """
    return text.replace('\u200d', '').split()


def sorani_join(tokens: list[str]) -> str:
    """Rejoin tokens into a string, suppressing spaces before punctuation."""
    if not tokens:
        return ""
    parts = [tokens[0]]
    for tok in tokens[1:]:
        if tok in _NO_SPACE_BEFORE:
            parts.append(tok)
        else:
            parts.append(' ')
            parts.append(tok)
    return ''.join(parts)
