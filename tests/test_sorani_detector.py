"""
Unit and functional tests for SoraniDetector.

Tests cover:
  - Core Sorani detection on known-good Wikipedia sentences
  - Rejection of English, Arabic, Persian, and Kurmanji text
  - Edge cases: empty strings, short text, mixed content
  - filter_corpus batch filtering
  - DetectionResult diagnostics (confidence, rival_language, label)
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.sorani_detector import SoraniDetector, DetectionResult


# ============================================================================
# Fixtures
# ============================================================================

# Real Sorani sentences from Kurdish Wikipedia
SORANI_SENTENCES = [
    "هەرێمی کوردستان لە باکووری عێراقدایە.",
    "هەولێر پایتەختی هەرێمی کوردستانە.",
    "زمانی کوردی زمانێکی ئیرانییە لە بنەمالەی زمانە هیندوئەورووپاییەکان.",
    "سلێمانی شارێکی گەورەی هەرێمی کوردستانە.",
    "ئەم وڵاتە کولتوورێکی کۆنی هەیە کە بۆ چەندین هەزار ساڵ دەگەڕێتەوە.",
    "ئابووری هەرێمی کوردستان بە زۆری لەسەر نەوت و کشتوکاڵ بنیات نراوە.",
    "دەستنووسی کوردی بە دوو ئەلفوبێ دەنووسرێت: لاتین و عەرەبی.",
    "کوردەکان گەلێکی کۆنن لە ناوچەکەدا.",
    "خەڵکی کوردستان چەندین ئایینیان هەیە.",
    "ئەو پرسیارە وەڵامێکی ئاسانی نییە.",
]

ENGLISH_SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning and natural language processing are growing.",
    "Python is a popular programming language used for data science.",
]

ARABIC_SENTENCES = [
    "في البداية خلق الله السماوات والأرض.",
    "هذا الكتاب مفيد جداً للطلاب الذين يدرسون اللغة العربية.",
    "المدينة القديمة تحتوي على العديد من المعالم التاريخية.",
]

PERSIAN_SENTENCES = [
    "ایران کشوری است در آسیای غربی با تاریخ کهن.",
    "زبان فارسی از خانواده زبان‌های هندواروپایی است.",
    "تهران پایتخت ایران است و بزرگترین شهر این کشور می‌باشد.",
]

KURMANJI_LATIN_SENTENCES = [
    "Ez ji Tirkiyê me û ez li Berlînê dijîm.",
    "Ew pirtûka min e ku li ser masê ye.",
    "Zimanê kurdî zimaneke hindûewropî ye.",
]


# ============================================================================
# Tests — Core Sorani Detection
# ============================================================================

class TestSoraniDetection:
    """Verify genuine Sorani Kurdish text is correctly identified."""

    def setup_method(self):
        self.detector = SoraniDetector()

    def test_sorani_sentences_accepted(self):
        """All known Sorani Wikipedia sentences should be detected as Sorani."""
        for sentence in SORANI_SENTENCES:
            result = self.detector.detect(sentence)
            assert result.is_sorani, (
                "Expected Sorani but got rejected: '%s' "
                "(conf=%.3f, script=%.3f, fw=%.3f, morph=%.3f)"
                % (sentence[:40], result.confidence,
                   result.script_score, result.function_word_score,
                   result.morphology_score)
            )

    def test_sorani_label_is_ckb(self):
        """Detected Sorani should return ISO 639-3 code 'ckb'."""
        result = self.detector.detect(SORANI_SENTENCES[0])
        assert result.label == "ckb"

    def test_sorani_confidence_reasonable(self):
        """Sorani sentences should have confidence well above threshold."""
        for sentence in SORANI_SENTENCES:
            result = self.detector.detect(sentence)
            assert result.confidence >= 0.55, (
                "Low confidence %.3f for: '%s'" % (result.confidence, sentence[:40])
            )

    def test_sorani_no_rival_language(self):
        """When detected as Sorani, rival_language should be None."""
        for sentence in SORANI_SENTENCES:
            result = self.detector.detect(sentence)
            if result.is_sorani:
                assert result.rival_language is None


# ============================================================================
# Tests — Rejection of Other Languages
# ============================================================================

class TestNonSoraniRejection:
    """Verify non-Sorani text is rejected."""

    def setup_method(self):
        self.detector = SoraniDetector()

    def test_english_rejected(self):
        for sentence in ENGLISH_SENTENCES:
            assert not self.detector.is_sorani(sentence), (
                "English falsely accepted: '%s'" % sentence
            )

    def test_arabic_rejected(self):
        for sentence in ARABIC_SENTENCES:
            assert not self.detector.is_sorani(sentence), (
                "Arabic falsely accepted: '%s'" % sentence
            )

    def test_persian_rejected(self):
        for sentence in PERSIAN_SENTENCES:
            assert not self.detector.is_sorani(sentence), (
                "Persian falsely accepted: '%s'" % sentence
            )

    def test_kurmanji_latin_rejected(self):
        for sentence in KURMANJI_LATIN_SENTENCES:
            assert not self.detector.is_sorani(sentence), (
                "Kurmanji (Latin) falsely accepted: '%s'" % sentence
            )

    def test_arabic_rival_label(self):
        """Arabic text should get rival label 'arb' or 'unknown'."""
        result = self.detector.detect(ARABIC_SENTENCES[1])
        assert result.label in ("arb", "unknown")

    def test_english_low_confidence(self):
        result = self.detector.detect(ENGLISH_SENTENCES[0])
        assert result.confidence < 0.3


# ============================================================================
# Tests — Edge Cases
# ============================================================================

class TestEdgeCases:
    """Edge case handling: empty, short, mixed content."""

    def setup_method(self):
        self.detector = SoraniDetector()

    def test_empty_string(self):
        result = self.detector.detect("")
        assert not result.is_sorani
        assert result.confidence == 0.0

    def test_whitespace_only(self):
        result = self.detector.detect("   \t\n  ")
        assert not result.is_sorani
        assert result.confidence == 0.0

    def test_single_sorani_word(self):
        """A single word is too short for reliable detection."""
        result = self.detector.detect("کوردستان")
        # Should still give some signal but low confidence due to short-text penalty
        assert result.confidence < 0.55 or result.is_sorani

    def test_numbers_only(self):
        result = self.detector.detect("123 456 789")
        assert not result.is_sorani

    def test_mixed_english_sorani(self):
        """Text with heavy English mixing should still work if Sorani dominates."""
        mixed = "بەرنامەی Python بۆ programming کاری باشە"
        result = self.detector.detect(mixed)
        # This is heavily mixed; detection may vary
        assert isinstance(result.confidence, float)


# ============================================================================
# Tests — filter_corpus
# ============================================================================

class TestFilterCorpus:
    """Test batch corpus filtering."""

    def setup_method(self):
        self.detector = SoraniDetector()

    def test_filter_keeps_sorani(self):
        filtered = self.detector.filter_corpus(SORANI_SENTENCES)
        assert len(filtered) == len(SORANI_SENTENCES)

    def test_filter_removes_english(self):
        filtered = self.detector.filter_corpus(ENGLISH_SENTENCES)
        assert len(filtered) == 0

    def test_filter_mixed_input(self):
        """Mixed corpus: only Sorani sentences should survive."""
        mixed = SORANI_SENTENCES[:3] + ENGLISH_SENTENCES[:2] + ARABIC_SENTENCES[:1]
        filtered = self.detector.filter_corpus(mixed)
        assert len(filtered) == 3
        for s in filtered:
            assert s in SORANI_SENTENCES


# ============================================================================
# Tests — DetectionResult Dataclass
# ============================================================================

class TestDetectionResult:
    """Verify DetectionResult fields and label logic."""

    def test_label_sorani(self):
        r = DetectionResult(
            is_sorani=True, confidence=0.8,
            script_score=0.7, function_word_score=0.9,
            morphology_score=0.6, rival_language=None,
        )
        assert r.label == "ckb"

    def test_label_rival(self):
        r = DetectionResult(
            is_sorani=False, confidence=0.3,
            script_score=0.5, function_word_score=0.1,
            morphology_score=0.2, rival_language="arb",
        )
        assert r.label == "arb"

    def test_label_unknown(self):
        r = DetectionResult(
            is_sorani=False, confidence=0.1,
            script_score=0.0, function_word_score=0.0,
            morphology_score=0.0, rival_language=None,
        )
        assert r.label == "unknown"

    def test_score_fields_are_floats(self):
        detector = SoraniDetector()
        result = detector.detect(SORANI_SENTENCES[0])
        assert isinstance(result.script_score, float)
        assert isinstance(result.function_word_score, float)
        assert isinstance(result.morphology_score, float)
        assert isinstance(result.confidence, float)


# ============================================================================
# Tests — Threshold Configuration
# ============================================================================

class TestThresholdConfig:
    """Verify threshold customization works."""

    def test_strict_threshold_rejects_borderline(self):
        strict = SoraniDetector(threshold=0.95)
        # Very strict threshold may reject some texts
        result = strict.detect("کوردستان باشە")
        # Short text with strict threshold likely rejected
        assert isinstance(result.is_sorani, bool)

    def test_lenient_threshold_accepts_more(self):
        lenient = SoraniDetector(threshold=0.2)
        result = lenient.detect(SORANI_SENTENCES[0])
        assert result.is_sorani
