"""
Sorani Kurdish Language Detector

Preprocessing filter that identifies whether a given text is Sorani Kurdish
(Central Kurdish / کوردیی ناوەندی), distinguishing it from:
  - Kurmanji (Northern Kurdish) — different function words, often Latin script
  - Arabic — no Kurdish-specific characters (ڕ, ڵ, ۆ, ێ)
  - Persian/Farsi — shares پ/چ/گ but lacks ڕ/ڵ, different function words
  - Other languages — script and vocabulary mismatch

Detection uses a weighted multi-signal scoring approach:
  1. Script analysis — Arabic-script ratio, Kurdish-only characters
  2. Function word matching — Sorani-specific vs Kurmanji-specific markers
  3. Morphological markers — verb prefixes (دە-), definiteness suffixes (-ەکە)
  4. Character frequency — ە distribution (very high in Sorani; diagnostic)

Sources:
  - Hassani (2018), Kurdish-BLARK — dialect identification features
  - Ahmadi (2020), KLPT — Kurdish language processing toolkit
  - Amin (2016), Verb Grammar of the Kurdish Language
"""

import re
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Character sets
# ---------------------------------------------------------------------------

# Characters found in Kurdish (both Sorani and Kurmanji-in-Arabic-script)
# but NOT in standard Arabic: ڕ ڵ ڤ ۆ ێ ە پ چ گ ژ
_KURDISH_CHARS = frozenset("ڕڵڤۆێەپچگژ")

# Characters unique to Kurdish — absent in both Arabic AND Persian.
# Persian has پ, چ, گ, ژ but not these:
_KURDISH_ONLY_CHARS = frozenset("ڕڵۆێ")

# Arabic script range (broad)
_ARABIC_SCRIPT_RE = re.compile(r"[\u0600-\u06FF\uFB50-\uFDFF\uFE70-\uFEFF]")

# ---------------------------------------------------------------------------
# Function words — Sorani vs Kurmanji vs Arabic vs Persian
# ---------------------------------------------------------------------------

# High-frequency Sorani function words (prepositions, pronouns, particles,
# conjunctions, demonstratives). These rarely appear in Kurmanji or Arabic.
_SORANI_FUNCTION_WORDS = frozenset({
    # Pronouns
    "من", "تۆ", "ئەو", "ئێمە", "ئێوە", "ئەوان",
    # Demonstratives
    "ئەم", "ئەمە", "ئەوە",
    # Prepositions
    "لە", "بۆ", "لەگەڵ", "بە", "لەسەر", "لەبەر", "لەژێر",
    "لەنێو", "لەپێش", "لەدوای", "تا", "بەبێ",
    # Particles & conjunctions
    "و", "کە", "بەڵام", "یان", "چونکە", "بۆیە", "ئەگەر",
    "هەر", "تەنها", "هەروەها",
    # Question words
    "کێ", "چی", "کەی", "چۆن", "کوێ", "چەند",
    # Copula / existential
    "هەیە", "نیە", "نییە",
    # Common adverbs/particles
    "زۆر", "کەم", "هیچ", "هەموو", "خۆ", "باش",
    # Verb-related
    "دەبێت", "دەکرێت", "ناکرێت", "نابێت",
})

# Kurmanji function words that are different from Sorani.
# If these appear, the text is likely Kurmanji, not Sorani.
_KURMANJI_FUNCTION_WORDS = frozenset({
    # Kurmanji pronouns (Latin or Arabic script)
    "ez", "tu", "ew", "em", "hûn", "ewan",
    # Kurmanji prepositions/particles
    "ji", "bi", "di", "li", "nav", "ser", "ber", "jêr",
    # Kurmanji conjunctions/particles
    "lê", "belê", "çimkî", "eger",
    # Kurmanji Latin-script markers
    "de", "ne", "ye", "in",
    # Kurmanji Arabic-script specific (different from Sorani)
    "ژی", "دی", "ناڤ",
})

# Arabic function words unlikely in Sorani Kurdish text.
_ARABIC_FUNCTION_WORDS = frozenset({
    "في", "من", "إلى", "على", "هذا", "هذه", "ذلك", "التي", "الذي",
    "أن", "لا", "ما", "هو", "هي", "كان", "عن", "مع", "قد",
    "بين", "ثم", "أو", "حتى", "إذا", "لم", "لن",
})
# Note: "من" appears in both Sorani and Arabic; it's excluded from
# Arabic-only scoring by checking alongside other Arabic markers.

# Persian function words that differ from Sorani.
_PERSIAN_FUNCTION_WORDS = frozenset({
    "است", "این", "آن", "را", "که", "با", "از", "به",
    "برای", "تا", "یا", "اگر", "ولی", "اما", "هم",
    "می", "نمی", "خواهد", "شد", "بود",
})

# ---------------------------------------------------------------------------
# Morphological markers — Sorani-specific affixes
# ---------------------------------------------------------------------------

# Present-tense verb prefixes (diagnostic for Sorani)
_SORANI_VERB_PREFIX_RE = re.compile(
    r"\b(?:دە|ئە|نا|بی?)\S{2,}"
)

# Definiteness markers (Sorani -ەکە/-ەکان, absent in Kurmanji)
_SORANI_DEFINITE_RE = re.compile(
    r"\S+(?:ەکە|یەکە|ەکان|یەکان)\b"
)

# Indefinite markers (Sorani -ێک/-ێکی)
_SORANI_INDEFINITE_RE = re.compile(
    r"\S+(?:ێک|ێکی)\b"
)

# Sorani ezafe marker (ی between noun-adjective; very frequent)
_SORANI_EZAFE_RE = re.compile(
    r"\S+ی\s+\S+"
)


# ---------------------------------------------------------------------------
# Detection result
# ---------------------------------------------------------------------------

@dataclass
class DetectionResult:
    """Result of Sorani language detection on a text."""
    is_sorani: bool
    confidence: float
    script_score: float
    function_word_score: float
    morphology_score: float
    rival_language: str | None

    @property
    def label(self) -> str:
        if self.is_sorani:
            return "ckb"  # ISO 639-3 for Central Kurdish (Sorani)
        if self.rival_language:
            return self.rival_language
        return "unknown"


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class SoraniDetector:
    """Detect whether text is Sorani Kurdish.

    Uses a weighted multi-signal approach combining script analysis,
    function word frequency, and morphological marker matching.

    Args:
        threshold: Minimum confidence score (0-1) to classify as Sorani.
            Default 0.55 balances recall vs precision for noisy web text.
        min_words: Minimum word count for reliable detection. Texts
            shorter than this get a reduced confidence penalty.
    """

    def __init__(self, threshold: float = 0.55, min_words: int = 2):
        self.threshold = threshold
        self.min_words = min_words

    def detect(self, text: str) -> DetectionResult:
        """Classify a text string as Sorani Kurdish or not.

        Returns a DetectionResult with confidence score and diagnostics.
        """
        text = text.strip()
        if not text:
            return DetectionResult(
                is_sorani=False, confidence=0.0,
                script_score=0.0, function_word_score=0.0,
                morphology_score=0.0, rival_language=None,
            )

        words = text.split()
        word_count = len(words)

        # --- Signal 1: Script analysis (weight 0.35) ---
        script_score, has_kurdish_only = self._score_script(text)

        # --- Signal 2: Function word matching (weight 0.40) ---
        fw_score, rival = self._score_function_words(words)

        # --- Signal 3: Morphological markers (weight 0.25) ---
        morph_score = self._score_morphology(text)

        # Weighted combination
        confidence = (
            0.35 * script_score
            + 0.40 * fw_score
            + 0.25 * morph_score
        )

        # Short-text penalty: reduce confidence for texts below min_words
        if word_count < self.min_words:
            confidence *= 0.7

        # Hard reject: no Kurdish-only characters at all in a long text
        if not has_kurdish_only and word_count > 5:
            confidence *= 0.5

        is_sorani = confidence >= self.threshold

        return DetectionResult(
            is_sorani=is_sorani,
            confidence=round(confidence, 4),
            script_score=round(script_score, 4),
            function_word_score=round(fw_score, 4),
            morphology_score=round(morph_score, 4),
            rival_language=rival if not is_sorani else None,
        )

    def is_sorani(self, text: str) -> bool:
        """Quick boolean check — returns True if text is likely Sorani."""
        return self.detect(text).is_sorani

    def filter_corpus(self, sentences: list[str]) -> list[str]:
        """Filter a list of sentences, keeping only Sorani Kurdish ones."""
        return [s for s in sentences if self.is_sorani(s)]

    # -------------------------------------------------------------------
    # Scoring components
    # -------------------------------------------------------------------

    @staticmethod
    def _score_script(text: str) -> tuple[float, bool]:
        """Score based on script composition.

        Returns (score, has_kurdish_only_chars).
        High score = text is predominantly Arabic-script with Kurdish chars.
        """
        non_space = text.replace(" ", "").replace("\t", "").replace("\n", "")
        total = len(non_space)
        if total == 0:
            return 0.0, False

        arabic_count = len(_ARABIC_SCRIPT_RE.findall(text))
        arabic_ratio = arabic_count / total

        # Check for Kurdish-specific characters
        kurdish_char_count = sum(1 for c in text if c in _KURDISH_CHARS)
        kurdish_only_count = sum(1 for c in text if c in _KURDISH_ONLY_CHARS)
        has_kurdish_only = kurdish_only_count > 0

        # Latin characters (signal for Kurmanji-Latin or English)
        latin_count = sum(1 for c in non_space if "A" <= c <= "z")
        latin_ratio = latin_count / total

        # Scoring
        score = 0.0

        # Arabic script presence (must be dominant)
        if arabic_ratio > 0.7:
            score += 0.4
        elif arabic_ratio > 0.5:
            score += 0.2

        # Kurdish-specific characters
        kurdish_ratio = kurdish_char_count / total
        if kurdish_ratio > 0.10:
            score += 0.35
        elif kurdish_ratio > 0.05:
            score += 0.25
        elif kurdish_ratio > 0.02:
            score += 0.15
        elif has_kurdish_only:
            score += 0.05

        # Kurdish-ONLY chars (not shared with Persian)
        if kurdish_only_count >= 3:
            score += 0.25
        elif kurdish_only_count >= 1:
            score += 0.15

        # Penalize high Latin ratio (likely English/Kurmanji-Latin)
        if latin_ratio > 0.3:
            score *= 0.3

        return min(score, 1.0), has_kurdish_only

    @staticmethod
    def _score_function_words(words: list[str]) -> tuple[float, str | None]:
        """Score based on function word presence.

        Returns (score, rival_language_if_detected).
        """
        if not words:
            return 0.0, None

        lower_words = set(words)
        word_count = len(words)

        sorani_hits = len(lower_words & _SORANI_FUNCTION_WORDS)
        kurmanji_hits = len(lower_words & _KURMANJI_FUNCTION_WORDS)
        arabic_hits = len(lower_words & _ARABIC_FUNCTION_WORDS)
        persian_hits = len(lower_words & _PERSIAN_FUNCTION_WORDS)

        # Normalize by word count (cap at 1.0)
        sorani_ratio = min(sorani_hits / max(word_count, 1), 1.0)
        kurmanji_ratio = min(kurmanji_hits / max(word_count, 1), 1.0)
        arabic_ratio = min(arabic_hits / max(word_count, 1), 1.0)
        persian_ratio = min(persian_hits / max(word_count, 1), 1.0)

        rival = None
        max_rival = max(kurmanji_ratio, arabic_ratio, persian_ratio)

        # When no function words from ANY language are found, return a
        # neutral score — absence of evidence is not evidence of absence.
        # Short content-word-only sentences (e.g. "هەولێر پایتەختی هەرێمی
        # کوردستانە") contain no function words at all.
        if sorani_hits == 0 and max_rival == 0:
            return 0.5, None

        # Sorani score: boosted by Sorani hits, penalized by rivals
        score = sorani_ratio * 3.0  # amplify (max ≈ 3.0 before clamping)

        if kurmanji_ratio > sorani_ratio and kurmanji_ratio > 0.05:
            score *= 0.3
            rival = "kmr"  # ISO 639-3 for Kurmanji
        elif arabic_ratio > sorani_ratio and arabic_ratio > 0.1:
            score *= 0.2
            rival = "arb"
        elif persian_ratio > sorani_ratio and persian_ratio > 0.1:
            score *= 0.2
            rival = "fas"
        elif max_rival > 0.15 and sorani_ratio < 0.05:
            score *= 0.4
            if kurmanji_ratio >= arabic_ratio and kurmanji_ratio >= persian_ratio:
                rival = "kmr"
            elif arabic_ratio >= persian_ratio:
                rival = "arb"
            else:
                rival = "fas"

        return min(score, 1.0), rival

    @staticmethod
    def _score_morphology(text: str) -> float:
        """Score based on Sorani-specific morphological markers."""
        score = 0.0

        # Present-tense verb prefixes (دە-, ئە-)
        verb_hits = len(_SORANI_VERB_PREFIX_RE.findall(text))
        if verb_hits >= 2:
            score += 0.4
        elif verb_hits >= 1:
            score += 0.25

        # Definiteness markers (-ەکە, -ەکان)
        def_hits = len(_SORANI_DEFINITE_RE.findall(text))
        if def_hits >= 1:
            score += 0.3

        # Indefinite markers (-ێک)
        indef_hits = len(_SORANI_INDEFINITE_RE.findall(text))
        if indef_hits >= 1:
            score += 0.15

        # ە frequency — Sorani uses ە far more than any other language.
        # In typical Sorani text, ە constitutes 8-15% of characters.
        non_space = text.replace(" ", "")
        if non_space:
            schwa_ratio = non_space.count("ە") / len(non_space)
            if schwa_ratio > 0.06:
                score += 0.15

        return min(score, 1.0)
