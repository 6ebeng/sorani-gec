"""Tests for the SoraniSanitizer module (CRIT-2)."""

import pytest
from src.data.sanitizer import SoraniSanitizer


@pytest.fixture
def sanitizer():
    return SoraniSanitizer()


class TestStripArtifacts:
    def test_removes_urls(self, sanitizer):
        text = "ئەم بابەتە https://example.com لێرە دەبینیت"
        result = sanitizer.strip_artifacts(text)
        assert "https://example.com" not in result
        assert "ئەم بابەتە" in result

    def test_removes_emails(self, sanitizer):
        text = "پەیوەندی بکە بە user@example.com بۆ زانیاری"
        result = sanitizer.strip_artifacts(text)
        assert "user@example.com" not in result

    def test_removes_citations(self, sanitizer):
        text = "کوردستان [1] ناوچەیەکە [citation needed] لە ڕۆژهەڵاتی ناوەڕاست"
        result = sanitizer.strip_artifacts(text)
        assert "[1]" not in result
        assert "[citation needed]" not in result

    def test_removes_wiki_templates(self, sanitizer):
        text = "{{سەرچاوەکان}} ئەم بابەتە پێویستی بە"
        result = sanitizer.strip_artifacts(text)
        assert "{{" not in result

    def test_removes_social_tags(self, sanitizer):
        text = "#کوردستان @user ئەم بابەتە"
        result = sanitizer.strip_artifacts(text)
        assert "#کوردستان" not in result
        assert "@user" not in result


class TestMojibake:
    def test_detects_mojibake(self, sanitizer):
        assert sanitizer.detect_mojibake("Ã¯Â»Â¿some text")

    def test_clean_text_not_mojibake(self, sanitizer):
        assert not sanitizer.detect_mojibake("من دەچم بۆ قوتابخانە")


class TestNonProse:
    def test_detects_digit_heavy_line(self, sanitizer):
        assert sanitizer.is_predominantly_non_prose("123 456 789 012 345 678")

    def test_normal_text_is_prose(self, sanitizer):
        assert not sanitizer.is_predominantly_non_prose("من دەچم بۆ قوتابخانە")

    def test_empty_is_non_prose(self, sanitizer):
        assert sanitizer.is_predominantly_non_prose("   ")


class TestScriptRatio:
    def test_arabic_script_passes(self, sanitizer):
        assert sanitizer.passes_script_ratio("من دەچم بۆ قوتابخانە")

    def test_mostly_latin_fails(self, sanitizer):
        assert not sanitizer.passes_script_ratio("This is English text with no Kurdish")

    def test_empty_fails(self, sanitizer):
        assert not sanitizer.passes_script_ratio("")


class TestLengthFilter:
    def test_too_short(self):
        s = SoraniSanitizer(min_tokens=5)
        assert not s.passes_length_filter("دوو وشە")

    def test_too_long(self):
        s = SoraniSanitizer(max_tokens=5)
        text = " ".join(["وشە"] * 10)
        assert not s.passes_length_filter(text)

    def test_within_range(self):
        s = SoraniSanitizer(min_tokens=2, max_tokens=10)
        assert s.passes_length_filter("من دەچم بۆ قوتابخانە")


class TestNearDuplicate:
    def test_exact_duplicate_detected(self, sanitizer):
        text = "من دەچم بۆ قوتابخانە هەر ڕۆژ"
        assert not sanitizer.is_near_duplicate(text)  # first time
        assert sanitizer.is_near_duplicate(text)  # exact repeat

    def test_different_text_not_duplicate(self, sanitizer):
        assert not sanitizer.is_near_duplicate("من دەچم بۆ قوتابخانە هەر ڕۆژ")
        assert not sanitizer.is_near_duplicate("ئەو کتێبەکەی خوێندەوە لە کتێبخانە")


class TestSanitizeLine:
    def test_clean_sorani_passes(self, sanitizer):
        text = "من دەچم بۆ قوتابخانە هەر ڕۆژ بە پیادە"
        result = sanitizer.sanitize_line(text)
        assert result is not None

    def test_url_stripped_and_passes(self, sanitizer):
        text = "سەردانی https://example.com بکە بۆ زانیاری زیاتر لەم بابەتە"
        result = sanitizer.sanitize_line(text)
        assert result is not None
        assert "https" not in result

    def test_mojibake_dropped(self, sanitizer):
        text = "Ã¯Â»Â¿ invalid encoding"
        result = sanitizer.sanitize_line(text)
        assert result is None

    def test_empty_dropped(self, sanitizer):
        assert sanitizer.sanitize_line("") is None
        assert sanitizer.sanitize_line("   ") is None


class TestSanitizeCorpus:
    def test_corpus_deduplication(self):
        sanitizer = SoraniSanitizer(min_tokens=3, near_dup_threshold=0.90)
        lines = [
            "من دەچم بۆ قوتابخانە هەر ڕۆژ بە پیادە",
            "من دەچم بۆ قوتابخانە هەر ڕۆژ بە پیادە",  # exact dup
            "ئەو کتێبەکەی خوێندەوە لە کتێبخانە بۆ ئامادەکاری",
        ]
        result = sanitizer.sanitize_corpus(lines)
        assert len(result) == 2  # one duplicate removed

    def test_stats_populated(self):
        sanitizer = SoraniSanitizer(min_tokens=3)
        lines = ["ab", "من دەچم بۆ قوتابخانە هەر ڕۆژ بە پیادە"]
        sanitizer.sanitize_corpus(lines)
        # "ab" should be dropped for being too short or low script ratio
        assert sum(sanitizer.stats.values()) >= 1


class TestExcessiveRepetition:
    def test_repeated_chars(self, sanitizer):
        assert sanitizer.has_excessive_repetition("ههههههههههههه")

    def test_normal_text(self, sanitizer):
        assert not sanitizer.has_excessive_repetition("من دەچم بۆ قوتابخانە")
