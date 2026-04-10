"""
Corpus Sanitizer for Sorani Kurdish Text

Sits between collection (01_collect_data.py) and normalization (02_normalize.py)
to remove noise that the normalizer is not designed to handle: URLs, citation
brackets, non-prose content, mojibake, and near-duplicates.

CRIT-2 in gap_analysis_and_missing_features.md.
"""

import hashlib
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------

# URLs (http, https, ftp, www)
_URL_RE = re.compile(
    r'https?://[^\s<>\"]+|ftp://[^\s<>\"]+|www\.[^\s<>\"]+',
    re.IGNORECASE,
)

# Email addresses
_EMAIL_RE = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')

# Citation/reference brackets: [1], [12], [note 3], [citation needed]
_CITATION_RE = re.compile(r'\[\s*(?:\d+|citation needed|note\s*\d*)\s*\]', re.IGNORECASE)

# Wiki-style templates: {{...}}
_TEMPLATE_RE = re.compile(r'\{\{[^}]*\}\}')

# Lines that are predominantly numbers/Latin (used for table/formula detection)
_ARABIC_SCRIPT_RE = re.compile(r'[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]')

# Mojibake signature characters (common when UTF-8 is misread as Latin-1)
_MOJIBAKE_RE = re.compile(r'[Ã¯Â»Â¿Ã‚Ã©Ã«Ã¨Ã¼Ã¶Ã¤ï»¿]{3,}')

# Repeated characters (3+ identical, excluding Arabic script connector chars)
_REPEATED_CHAR_RE = re.compile(r'(.)\1{4,}')

# Hash tag / social media artifacts
_SOCIAL_RE = re.compile(r'(?:#\w+|@\w+)')

# Arabic diacritics range for combining-class validation (CRIT-1)
_ARABIC_DIACRITICS = set(range(0x064B, 0x0660))  # U+064B..U+065F

# Common Sorani abbreviations that should not be dropped by the length filter
SORANI_ABBREVIATIONS = {
    "د.",   # doctor / professor
    "ب.",   # section / paragraph
    "پ.",   # professor
    "م.",   # mister / engineer
    "ژ.",   # number
    "ل.",   # page
    "هتد.", # etc.
    "بڕ.",  # paragraph
}

# Maximum fingerprint cache size for near-duplicate detection
_MAX_FINGERPRINT_CACHE = 500_000


class SoraniSanitizer:
    """Cleans raw text before normalization.

    Each method is a filter that returns the cleaned text or ``None``
    to signal that the line should be dropped entirely.
    """

    def __init__(
        self,
        min_kurdish_ratio: float = 0.50,
        max_latin_ratio: float = 0.30,
        min_tokens: int = 3,
        max_tokens: int = 200,
        near_dup_threshold: float = 0.90,
        sorani_detector: Optional[object] = None,
    ):
        self.min_kurdish_ratio = min_kurdish_ratio
        self.max_latin_ratio = max_latin_ratio
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.near_dup_threshold = near_dup_threshold
        self._detector = sorani_detector
        # Fingerprint cache for near-duplicate detection (char 3-gram sets)
        self._fingerprints: list[frozenset[str]] = []
        # Stats
        self.stats: dict[str, int] = {}

    def _bump(self, reason: str) -> None:
        self.stats[reason] = self.stats.get(reason, 0) + 1

    # ------------------------------------------------------------------
    # Individual filters
    # ------------------------------------------------------------------

    def strip_artifacts(self, text: str) -> str:
        """Remove URLs, emails, citation brackets, templates, social tags."""
        text = _URL_RE.sub('', text)
        text = _EMAIL_RE.sub('', text)
        text = _CITATION_RE.sub('', text)
        text = _TEMPLATE_RE.sub('', text)
        text = _SOCIAL_RE.sub('', text)
        # Collapse leftover whitespace
        text = re.sub(r'\s{2,}', ' ', text).strip()
        return text

    def detect_mojibake(self, text: str) -> bool:
        """Return True if the text shows signs of encoding corruption."""
        return bool(_MOJIBAKE_RE.search(text))

    def has_malformed_diacritics(self, text: str) -> bool:
        """Detect malformed Arabic diacritics combining with wrong base characters.

        Returns True if a diacritic (U+064B-U+065F) appears after a non-Arabic
        base character or if two diacritics of the same combining class stack
        on one base (invalid in Unicode canonical ordering).
        """
        import unicodedata
        prev_is_arabic_base = False
        prev_combining_class = 0
        for ch in text:
            cp = ord(ch)
            if cp in _ARABIC_DIACRITICS:
                cc = unicodedata.combining(ch)
                if not prev_is_arabic_base:
                    return True
                if cc != 0 and cc == prev_combining_class:
                    return True
                prev_combining_class = cc
            else:
                prev_is_arabic_base = bool(_ARABIC_SCRIPT_RE.match(ch))
                prev_combining_class = 0
        return False

    def is_predominantly_non_prose(self, text: str) -> bool:
        """Return True if text looks like a table row, formula, or metadata."""
        non_space = re.sub(r'\s', '', text)
        if not non_space:
            return True
        arabic_count = len(_ARABIC_SCRIPT_RE.findall(non_space))
        arabic_ratio = arabic_count / len(non_space)
        if arabic_ratio < 0.30:
            return True
        # High digit ratio → likely a table or formula
        digit_count = sum(1 for ch in non_space if ch.isdigit())
        if len(non_space) > 5 and digit_count / len(non_space) > 0.50:
            return True
        return False

    def passes_language_filter(self, text: str) -> bool:
        """Re-apply Sorani detection. Returns True if text is Sorani."""
        if self._detector is None:
            return True
        result = self._detector.detect(text)
        return result.is_sorani

    def passes_length_filter(self, text: str) -> bool:
        """Check min/max token count.

        Sentences consisting solely of known Sorani abbreviations bypass
        the minimum-token filter to avoid losing abbreviation-only lines.
        """
        tokens = text.split()
        if len(tokens) < self.min_tokens:
            # Allow if all tokens are known abbreviations
            if all(t in SORANI_ABBREVIATIONS for t in tokens):
                return True
            return False
        return len(tokens) <= self.max_tokens

    def passes_script_ratio(self, text: str) -> bool:
        """Check that enough of the text is Arabic script."""
        non_space = re.sub(r'\s', '', text)
        if not non_space:
            return False
        arabic_count = len(_ARABIC_SCRIPT_RE.findall(non_space))
        ratio = arabic_count / len(non_space)
        if ratio < self.min_kurdish_ratio:
            return False
        # Also check Latin ratio
        latin_count = sum(1 for ch in non_space if 'A' <= ch <= 'Z' or 'a' <= ch <= 'z')
        if len(non_space) > 0 and latin_count / len(non_space) > self.max_latin_ratio:
            return False
        return True

    def has_excessive_repetition(self, text: str) -> bool:
        """Detect lines with excessive character repetition (spam/noise)."""
        return bool(_REPEATED_CHAR_RE.search(text))

    def is_near_duplicate(self, text: str) -> bool:
        """Check text against previously seen sentences using char 3-gram Jaccard.

        Returns True if a near-duplicate exists (above threshold).
        Evicts oldest fingerprints when cache exceeds _MAX_FINGERPRINT_CACHE.
        """
        trigrams = self._char_trigrams(text)
        if not trigrams:
            return False
        for seen in self._fingerprints:
            if not seen:
                continue
            jaccard = len(trigrams & seen) / len(trigrams | seen)
            if jaccard >= self.near_dup_threshold:
                return True
        # LRU eviction: drop oldest entries when cache is full
        if len(self._fingerprints) >= _MAX_FINGERPRINT_CACHE:
            self._fingerprints = self._fingerprints[len(self._fingerprints) // 4:]
        self._fingerprints.append(trigrams)
        return False

    @staticmethod
    def _char_trigrams(text: str) -> frozenset[str]:
        """Extract character-level 3-grams as a frozenset."""
        text = re.sub(r'\s+', ' ', text.strip())
        if len(text) < 3:
            return frozenset()
        return frozenset(text[i:i + 3] for i in range(len(text) - 2))

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def sanitize_line(self, text: str) -> Optional[str]:
        """Run all filters on a single line. Returns cleaned text or None."""
        # Strip artifacts first
        text = self.strip_artifacts(text)
        if not text:
            self._bump("empty_after_strip")
            return None

        if self.detect_mojibake(text):
            self._bump("mojibake")
            return None

        if self.has_malformed_diacritics(text):
            self._bump("malformed_diacritics")
            return None

        if self.is_predominantly_non_prose(text):
            self._bump("non_prose")
            return None

        if self.has_excessive_repetition(text):
            self._bump("excessive_repetition")
            return None

        if not self.passes_script_ratio(text):
            self._bump("low_script_ratio")
            return None

        if not self.passes_length_filter(text):
            tokens = text.split()
            if len(tokens) < self.min_tokens:
                self._bump("too_short")
            else:
                self._bump("too_long")
            return None

        if not self.passes_language_filter(text):
            self._bump("not_sorani")
            return None

        if self.is_near_duplicate(text):
            self._bump("near_duplicate")
            return None

        return text

    def sanitize_corpus(self, lines: list[str]) -> list[str]:
        """Sanitize all lines and return the survivors."""
        self.stats = {}
        self._fingerprints = []
        result = []
        for line in lines:
            cleaned = self.sanitize_line(line.strip())
            if cleaned is not None:
                result.append(cleaned)
        return result
