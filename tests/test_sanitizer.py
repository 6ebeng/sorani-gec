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

    def test_removes_page_markers(self, sanitizer):
        text = "--- Page ۲۳ --- بۆنموونە ئەم وشەیە دروستە"
        result = sanitizer.strip_artifacts(text)
        assert "Page" not in result
        assert "---" not in result
        assert "بۆنموونە" in result

    def test_removes_page_markers_western_digits(self, sanitizer):
        text = "--- Page 88 --- ئەم ڕستەیە لە کتێبەکەیە"
        result = sanitizer.strip_artifacts(text)
        assert "Page" not in result

    def test_removes_page_markers_mid_sentence(self, sanitizer):
        text = "وشەی سادە شیکار ناکرێت --- Page ۸۹ --- هێڵکاریی دەخریتە ڕوو"
        result = sanitizer.strip_artifacts(text)
        assert "Page" not in result
        assert "شیکار" in result
        assert "هێڵکاریی" in result

    def test_removes_latex_math(self, sanitizer):
        text = "$N^{=e}۱ + V^e$ ئەگەر کرداری ڕستەکە ڕانەبردووی تێنەپەڕ بێت"
        result = sanitizer.strip_artifacts(text)
        assert "$" not in result
        assert "N^" not in result
        assert "ئەگەر" in result


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
        text = "من دەچم بۆ قوتابخانە هەر ڕۆژ بە پیادە."
        result = sanitizer.sanitize_line(text)
        assert result is not None

    def test_url_stripped_and_passes(self, sanitizer):
        text = "سەردانی https://example.com بکە بۆ زانیاری زیاتر لەم بابەتە."
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
            "من دەچم بۆ قوتابخانە هەر ڕۆژ بە پیادە.",
            "من دەچم بۆ قوتابخانە هەر ڕۆژ بە پیادە.",  # exact dup
            "ئەو کتێبەکەی خوێندەوە لە کتێبخانە بۆ ئامادەکاری.",
        ]
        result = sanitizer.sanitize_corpus(lines)
        assert len(result) == 2  # one duplicate removed

    def test_stats_populated(self):
        sanitizer = SoraniSanitizer(min_tokens=3)
        lines = ["ab", "من دەچم بۆ قوتابخانە هەر ڕۆژ بە پیادە."]
        sanitizer.sanitize_corpus(lines)
        # "ab" should be dropped for being too short or low script ratio
        assert sum(sanitizer.stats.values()) >= 1


class TestExcessiveRepetition:
    def test_repeated_chars(self, sanitizer):
        assert sanitizer.has_excessive_repetition("ههههههههههههه")

    def test_normal_text(self, sanitizer):
        assert not sanitizer.has_excessive_repetition("من دەچم بۆ قوتابخانە")


class TestGeminiArtifacts:
    def test_detects_wait_prefix(self, sanitizer):
        assert sanitizer.has_gemini_artifacts('Wait, Item ۳: "سێیەم - دەرکەوت"')

    def test_detects_check_prefix(self, sanitizer):
        assert sanitizer.has_gemini_artifacts('* *Check Item ۱۰:* "غازی فاتیح"')

    def test_detects_lets_prefix(self, sanitizer):
        assert sanitizer.has_gemini_artifacts("Let's look at the ڕ and ڵ again")

    def test_normal_kurdish_no_match(self, sanitizer):
        assert not sanitizer.has_gemini_artifacts("من دەچم بۆ قوتابخانە هەر ڕۆژ")

    def test_sanitize_drops_gemini_line(self, sanitizer):
        text = 'Wait, Item ۳: "سێیەم دەرکەوت جەگەر خوێنیش لە کتیبی دەستووری زمانی کوردی"'
        result = sanitizer.sanitize_line(text)
        assert result is None


class TestPageMarkerRemoval:
    def test_sanitize_strips_page_marker(self, sanitizer):
        text = "--- Page ۲۳ --- بۆنموونە ئەم وشەیە بەکار دێت لە ڕێزمانی کوردی."
        result = sanitizer.sanitize_line(text)
        assert result is not None
        assert "Page" not in result

    def test_latex_stripped_in_sanitize(self, sanitizer):
        text = "$V^e$ ئەگەر کرداری ڕستەکە ڕانەبردووی تێنەپەڕ بێت لە زمانی کوردی."
        result = sanitizer.sanitize_line(text)
        assert result is not None
        assert "$" not in result


class TestStripListMarkers:
    def test_strips_arabic_numeral_dash(self, sanitizer):
        text = "۲- پێشناو وەک پۆلێکی ڕێزمانی دایە."
        result = sanitizer.strip_artifacts(text)
        assert not result.startswith("۲")
        assert "پێشناو" in result

    def test_strips_arabic_numeral_paren(self, sanitizer):
        text = "۵) یەکێکی دیکە لە جیاوازییەکانی زمانەکەیە."
        result = sanitizer.strip_artifacts(text)
        assert not result.startswith("۵")

    def test_strips_letter_paren(self, sanitizer):
        text = "ج) وەچەپێکەاتە دەتوانرێت بقرتێنرێت."
        result = sanitizer.strip_artifacts(text)
        assert not result.startswith("ج")

    def test_strips_leading_tatweel(self, sanitizer):
        text = "ـ ئاڵا لە دایک نەبووبوو."
        result = sanitizer.strip_artifacts(text)
        assert not result.startswith("ـ")
        assert "ئاڵا" in result

    def test_strips_superscript_footnote(self, sanitizer):
        text = "شیاو بێت لە هەر جێگەیەکی ڕستەکەدا بێت²."
        result = sanitizer.strip_artifacts(text)
        assert "²" not in result


class TestFormulaAndFragmentDetection:
    def test_has_formula_arrow(self, sanitizer):
        assert sanitizer.has_formula_notation("ڕستە ← ڕستەی هەواڵیی")

    def test_has_formula_equals(self, sanitizer):
        assert sanitizer.has_formula_notation('" کێ = " ی فارسی')

    def test_no_formula_clean_text(self, sanitizer):
        assert not sanitizer.has_formula_notation("من دەچم بۆ قوتابخانە.")

    def test_is_fragment_no_terminal_punct(self, sanitizer):
        assert sanitizer.is_fragment("تەرازووی زانستی زمانەوە کە وشە")

    def test_is_fragment_ends_with_colon(self, sanitizer):
        assert sanitizer.is_fragment("فۆنەتیکی و فڕێدانی هەندێ دەنگ، وەگ :")

    def test_is_fragment_ends_with_ellipsis(self, sanitizer):
        assert sanitizer.is_fragment("یان ڕابردووی بەردەوام، یان ڕابردووی...")

    def test_not_fragment_with_period(self, sanitizer):
        assert not sanitizer.is_fragment("ئەم یاسایە زیاتر لە ڕستەدا ڕوودەدات.")


class TestSplitLongSentence:
    def test_short_sentence_unchanged(self):
        text = "من دەچم بۆ قوتابخانە"
        result = SoraniSanitizer.split_long_sentence(text, max_tokens=50)
        assert result == [text]

    def test_splits_at_comma(self):
        # Build a sentence with two clauses > 5 tokens each, joined by ،
        clause1 = "کتێبەکەم خوێندەوە لە قوتابخانە"
        clause2 = "ئەو کتێبەکەی خوێندەوە لە ماڵەوە"
        text = f"{clause1}، {clause2}"
        result = SoraniSanitizer.split_long_sentence(text, max_tokens=5)
        assert len(result) == 2
        assert clause1 + "،" in result[0]

    def test_splits_at_period(self):
        clause1 = " ".join(["وشە"] * 8) + "."
        clause2 = " ".join(["وشە"] * 8) + "."
        text = f"{clause1} {clause2}"
        result = SoraniSanitizer.split_long_sentence(text, max_tokens=10)
        assert len(result) == 2

    def test_no_punctuation_returns_original(self):
        text = " ".join(["وشە"] * 20)
        result = SoraniSanitizer.split_long_sentence(text, max_tokens=10)
        assert len(result) == 1
        assert result[0] == text

    def test_greedy_merge_keeps_fragments_under_limit(self):
        # 3 short clauses joined by ،
        parts = ["وشە وشە وشە،", "وشە وشە،", "وشە وشە وشە وشە"]
        text = " ".join(parts)
        result = SoraniSanitizer.split_long_sentence(text, max_tokens=6)
        for r in result:
            assert len(r.split()) <= 6 or len(_get_fragments(text)) <= 1


def _get_fragments(text):
    """Helper to check Fragment splitting."""
    import re
    return re.split(r'(?<=[.،؛;:!?؟۔])\s+', text)
