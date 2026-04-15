"""
Sorani Kurdish Text Normalizer

Handles Arabic-script normalization for Sorani Kurdish text:
- Character normalization (e.g., different forms of ی, ک, ە)
- Whitespace normalization
- Zero-width character handling
- Optional diacritics removal
"""

import re
import unicodedata
from pathlib import Path
from typing import Optional


# ============================================================================
# Sorani Kurdish Character Mappings
# ============================================================================

# Standard Sorani Kurdish alphabet (Arabic script):
# ئ ا ب پ ت ج چ ح خ د ر ڕ ز ژ س ش ع غ ف ڤ ق ک گ ل ڵ م ن و ۆ ھ ە ی ێ

# Common normalization mappings
CHAR_NORMALIZATIONS = {
    # Kaf variants → Kurdish Kaf
    "\u0643": "\u06A9",  # Arabic Kaf → Kurdish Kaf (ک)
    
    # Yeh variants → Kurdish Yeh
    "\u064A": "\u06CC",  # Arabic Yeh → Kurdish Yeh (ی)
    "\u0649": "\u06CC",  # Alef Maksura → Kurdish Yeh
    
    # Tatweel (kashida) — cosmetic stretcher with no linguistic meaning.
    # Must be stripped before downstream token counting and pattern matching.
    "\u0640": "",  # Tatweel (ـ) → removed
    
    # Heh (U+0647) handled separately in normalize() — context-dependent.
    # Word-initial ه is consonant /h/; non-initial ه is vowel ə → ە (U+06D5).
    
    # Western digits → Extended Arabic-Indic digits (Sorani Kurdish uses Extended form U+06F0–U+06F9)
    "0": "\u06F0", "1": "\u06F1", "2": "\u06F2", "3": "\u06F3",
    "4": "\u06F4", "5": "\u06F5", "6": "\u06F6", "7": "\u06F7",
    "8": "\u06F8", "9": "\u06F9",
    
    # Standard Arabic-Indic digits → Extended Arabic-Indic digits
    "\u0660": "\u06F0", "\u0661": "\u06F1", "\u0662": "\u06F2", "\u0663": "\u06F3",
    "\u0664": "\u06F4", "\u0665": "\u06F5", "\u0666": "\u06F6", "\u0667": "\u06F7",
    "\u0668": "\u06F8", "\u0669": "\u06F9",
}

# Zero-width characters to remove.
# PIPE-25 design note: ZWNJ (U+200C) is preserved by default during
# normalization (preserve_zwnj=True) because it marks morpheme boundaries
# in compound verbs (e.g. دە‌چم) and is significant for display rendering.
# Downstream tokenizers (sorani_tokenize) strip ZWNJ for analysis purposes.
# This two-phase approach is intentional: normalized text retains the
# morpheme marker for human readability; tokenized text strips it so that
# token comparison is not affected by invisible characters.
ZERO_WIDTH_CHARS = [
    "\u200B",  # Zero-width space
    "\u200C",  # Zero-width non-joiner (ZWNJ) — keep selectively
    "\u200D",  # Zero-width joiner (ZWJ)
    "\u200E",  # Left-to-right mark
    "\u200F",  # Right-to-left mark
    "\uFEFF",  # BOM / Zero-width no-break space
]

# Diacritics (tashkeel) — optional removal
DIACRITICS = [
    "\u064B",  # Fathatan
    "\u064C",  # Dammatan
    "\u064D",  # Kasratan
    "\u064E",  # Fatha
    "\u064F",  # Damma
    "\u0650",  # Kasra
    "\u0651",  # Shadda
    "\u0652",  # Sukun
    "\u0670",  # Superscript Alef
]

# Regex for context-dependent Arabic Heh (U+0647) normalization.
# Matches U+0647 only when preceded by an Arabic-script character,
# indicating non-initial (medial/final) position within a word.
# Word-initial U+0647 (consonant /h/) is not matched and stays.
_NON_INITIAL_HEH = re.compile(
    r'(?<=[\u0621-\u063A\u0641-\u064A\u066E-\u06D4'
    r'\u06D5-\u06EF\u06FA-\u06FF\u0750-\u077F])\u0647'
)

# ============================================================================
# Parenthetical & Punctuation Normalization
# (standalone functions, applied after character normalization)
# ============================================================================

# Standalone parenthetical: (content) followed by space, punctuation, or EOS.
# NOT matched when ) is immediately followed by an Arabic-script letter,
# because that signals a clitic/suffix fused to the parenthetical:
#   (بەنو هاشم)ە   (قەرە یوسفی)تورکمانی   (شێوەکە)یە
_STANDALONE_PAREN_RE = re.compile(
    r'\([^)]+\)(?=[\s.،؟!\u06D4:;]|$)'
)


def strip_standalone_parentheticals(text: str) -> str:
    """Remove parenthetical content that is not morphologically bound.

    Strips (content) when the closing paren is followed by whitespace,
    punctuation, or end-of-string.  Keeps (content)clitic where the
    closing paren is immediately followed by an Arabic-script letter.
    """
    # Iterate because consecutive parens like (A)(B). leave (A) standalone
    # once (B) is removed in a single pass.
    prev = None
    while text != prev:
        prev = text
        text = _STANDALONE_PAREN_RE.sub('', text)
    # Collapse double spaces left by removal
    text = re.sub(r' {2,}', ' ', text).strip()
    return text


# Remove space(s) before punctuation: . ، ؟ ! ۔ : ;
_SPACE_BEFORE_PUNCT = re.compile(r'\s+([.،؟!۔:;])')
# Ensure space after Arabic comma when directly followed by Arabic letter
_COMMA_NO_SPACE = re.compile(r'،(?=[\u0600-\u06FF])')
# Ensure space after colon when directly followed by Arabic letter
_COLON_NO_SPACE = re.compile(r':(?=[\u0600-\u06FF])')
# Ensure space after ellipsis when directly followed by Arabic letter
_ELLIPSIS_NO_SPACE = re.compile(r'…(?=[\u0600-\u06FF])')


def normalize_punctuation(text: str) -> str:
    """Normalize punctuation spacing for Sorani Kurdish text.

    Fixes common OCR and digitisation artefacts:
    - Latin comma (,) → Arabic comma (،)
    - Removes spurious space before . ، ؟ ! ۔ : ;
    - Ensures space after ، and : when followed by Arabic letter
    - Normalises double period (..) → single and triple (...) → ellipsis
    - Ensures space after ellipsis (…) when followed by Arabic letter
    """
    # Latin comma → Arabic comma
    text = text.replace(',', '،')

    # Remove space(s) before punctuation marks
    text = _SPACE_BEFORE_PUNCT.sub(r'\1', text)

    # Collapse repeated commas (،، → ،) created by space removal
    text = re.sub(r'،{2,}', '،', text)

    # Ensure space after comma if followed by Arabic letter
    text = _COMMA_NO_SPACE.sub('، ', text)

    # Ensure space after colon if followed by Arabic letter
    text = _COLON_NO_SPACE.sub(': ', text)

    # Triple dots → Unicode ellipsis character (MUST come before double-dot fix)
    text = text.replace('...', '…')

    # Collapse any remaining run of 2+ consecutive periods to a single period
    text = re.sub(r'\.{2,}', '.', text)

    # Remove period immediately after ؟ or ! (redundant terminal mark)
    text = re.sub(r'([؟!])\.', r'\1', text)

    # Ensure space after ellipsis if followed by Arabic letter
    text = _ELLIPSIS_NO_SPACE.sub('… ', text)

    return text


class SoraniNormalizer:
    """Normalize Sorani Kurdish text for consistent processing."""
    
    def __init__(
        self,
        normalize_chars: bool = True,
        remove_diacritics: bool = False,
        remove_zero_width: bool = True,
        preserve_zwnj: bool = True,
        normalize_whitespace: bool = True,
    ):
        self.normalize_chars = normalize_chars
        self.remove_diacritics = remove_diacritics
        self.remove_zero_width = remove_zero_width
        self.preserve_zwnj = preserve_zwnj
        self.normalize_whitespace = normalize_whitespace
        
        # Build translation table
        self._build_translation_table()
    
    def _build_translation_table(self):
        """Build character translation table for fast normalization."""
        mapping = {}
        
        if self.normalize_chars:
            for src, dst in CHAR_NORMALIZATIONS.items():
                mapping[ord(src)] = dst
        
        if self.remove_diacritics:
            for char in DIACRITICS:
                mapping[ord(char)] = ""
        
        if self.remove_zero_width:
            for char in ZERO_WIDTH_CHARS:
                if self.preserve_zwnj and char == "\u200C":
                    continue
                mapping[ord(char)] = ""
        
        self._trans_table = str.maketrans(mapping)
    
    def normalize(self, text: str) -> str:
        """Apply full normalization pipeline to text."""
        if not text:
            return text
        
        # Unicode NFC normalization
        text = unicodedata.normalize("NFC", text)
        
        # Character-level normalization
        text = text.translate(self._trans_table)
        
        # Context-dependent Arabic Heh (U+0647) normalization:
        # Word-initial ه is consonant /h/ → preserved as U+0647
        # Non-initial ه (preceded by Arabic letter) is vowel ə → ە (U+06D5)
        if self.normalize_chars:
            text = _NON_INITIAL_HEH.sub("\u06D5", text)
        
        # Whitespace normalization
        if self.normalize_whitespace:
            text = self._normalize_whitespace(text)
        
        return text.strip()
    
    def normalize_with_offsets(self, text: str) -> tuple[str, list[int]]:
        """Normalize text and return a character offset map.

        Returns:
            (normalized_text, offset_map) where offset_map[i] gives the
            original-text character index that produced normalized_text[i].
            Useful for adjusting annotation spans after normalization.
        """
        if not text:
            return text, []

        # Build per-character mapping through each transform step.
        # We track which original index each current character came from.
        offsets = list(range(len(text)))

        # --- NFC normalization ---
        nfc = unicodedata.normalize("NFC", text)
        # NFC may change length (combining chars merge). Rebuild offsets
        # by character-by-character NFC alignment rather than proportional
        # approximation (PIPE-23 fix).
        if len(nfc) != len(text):
            new_offsets: list[int] = []
            src_idx = 0
            for nfc_idx in range(len(nfc)):
                new_offsets.append(offsets[min(src_idx, len(offsets) - 1)])
                # Advance src_idx past all original characters that
                # compose into this single NFC character.  We detect the
                # boundary by NFC-normalizing successive slices.
                src_idx += 1
                while src_idx < len(text):
                    # Check if the substring text[src_idx_start:src_idx] NFC-normalizes
                    # to exactly the NFC characters we've consumed so far.
                    probe = unicodedata.normalize("NFC", text[:src_idx])
                    if len(probe) > nfc_idx + 1:
                        break
                    src_idx += 1
            offsets = new_offsets
        text = nfc

        # --- Character-level translation ---
        new_text: list[str] = []
        new_offsets: list[int] = []
        for i, ch in enumerate(text):
            mapped = self._trans_table.get(ord(ch))
            if mapped is None:
                new_text.append(ch)
                new_offsets.append(offsets[i])
            elif isinstance(mapped, int):
                new_text.append(chr(mapped))
                new_offsets.append(offsets[i])
            elif isinstance(mapped, str):
                for c in mapped:
                    new_text.append(c)
                    new_offsets.append(offsets[i])
            # mapped == "" (deletion) → skip
        text = "".join(new_text)
        offsets = new_offsets

        # --- Context-dependent Heh ---
        if self.normalize_chars:
            result_text: list[str] = []
            result_offsets: list[int] = []
            pos = 0
            for m in _NON_INITIAL_HEH.finditer(text):
                # Copy everything before the match
                for j in range(pos, m.start()):
                    result_text.append(text[j])
                    result_offsets.append(offsets[j])
                # Replace matched heh
                result_text.append("\u06D5")
                result_offsets.append(offsets[m.start()])
                pos = m.end()
            for j in range(pos, len(text)):
                result_text.append(text[j])
                result_offsets.append(offsets[j])
            text = "".join(result_text)
            offsets = result_offsets

        # --- Whitespace normalization ---
        if self.normalize_whitespace:
            text, offsets = self._normalize_whitespace_with_offsets(text, offsets)

        # --- Strip ---
        stripped = text.strip()
        if len(stripped) < len(text):
            lead = len(text) - len(text.lstrip())
            offsets = offsets[lead:lead + len(stripped)]
        text = stripped

        return text, offsets
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize various whitespace characters."""
        # Replace multiple spaces with single space
        text = re.sub(r"[ \t]+", " ", text)
        # Normalize line endings
        text = re.sub(r"\r\n?", "\n", text)
        # Remove multiple blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text
    
    def _normalize_whitespace_with_offsets(
        self, text: str, offsets: list[int],
    ) -> tuple[str, list[int]]:
        """Whitespace normalization preserving an offset map."""
        result: list[str] = []
        result_offsets: list[int] = []
        i = 0
        while i < len(text):
            ch = text[i]
            if ch in (" ", "\t"):
                # Collapse consecutive spaces/tabs into one space
                result.append(" ")
                result_offsets.append(offsets[i])
                while i < len(text) and text[i] in (" ", "\t"):
                    i += 1
            elif ch == "\r":
                # Normalize \r\n or lone \r to \n
                result.append("\n")
                result_offsets.append(offsets[i])
                i += 1
                if i < len(text) and text[i] == "\n":
                    i += 1
            else:
                result.append(ch)
                result_offsets.append(offsets[i])
                i += 1
        # Collapse 3+ consecutive newlines to 2
        final: list[str] = []
        final_offsets: list[int] = []
        nl_count = 0
        for j, c in enumerate(result):
            if c == "\n":
                nl_count += 1
                if nl_count <= 2:
                    final.append(c)
                    final_offsets.append(result_offsets[j])
            else:
                nl_count = 0
                final.append(c)
                final_offsets.append(result_offsets[j])
        return "".join(final), final_offsets
    
    def normalize_file(self, input_path: str, output_path: str) -> int:
        """Normalize a text file line by line. Returns number of lines processed."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with open(input_path, "r", encoding="utf-8") as fin, \
             open(output_path, "w", encoding="utf-8") as fout:
            for line in fin:
                normalized = self.normalize(line.rstrip("\n"))
                if normalized:  # Skip empty lines
                    fout.write(normalized + "\n")
                    count += 1
        return count


def sentence_split(text: str) -> list[str]:
    """Split Sorani text into sentences.
    
    Handles both Arabic/Kurdish sentence-ending markers and standard punctuation.
    """
    # Kurdish/Arabic sentence boundaries: ۔ (Arabic full stop), . ! ? ؟
    sentences = re.split(r'(?<=[.!?؟۔])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def deduplicate_sentences(sentences: list[str]) -> list[str]:
    """Remove duplicate sentences while preserving order."""
    seen = set()
    result = []
    for s in sentences:
        normalized = s.strip().lower()
        if normalized not in seen:
            seen.add(normalized)
            result.append(s)
    return result


if __name__ == "__main__":
    # Quick test
    normalizer = SoraniNormalizer()
    
    test_texts = [
        "سڵاو، ئەم تێستێکە بۆ نۆرماڵکردنی تێکست.",
        "کوردستان   زۆر   جوانە",  # Extra spaces
    ]
    
    for text in test_texts:
        result = normalizer.normalize(text)
        print(f"Input:  {text}")
        print(f"Output: {result}")
        print()
