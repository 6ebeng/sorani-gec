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
    
    # Heh (U+0647) handled separately in normalize() — context-dependent.
    # Word-initial ه is consonant /h/; non-initial ه is vowel ə → ە (U+06D5).
    
    # Western digits → Arabic-Indic digits (Sorani Kurdish uses Arabic-Indic)
    "0": "\u0660", "1": "\u0661", "2": "\u0662", "3": "\u0663",
    "4": "\u0664", "5": "\u0665", "6": "\u0666", "7": "\u0667",
    "8": "\u0668", "9": "\u0669",
    
    # Extended Arabic-Indic digits → Standard Arabic-Indic digits
    "\u06F0": "\u0660", "\u06F1": "\u0661", "\u06F2": "\u0662", "\u06F3": "\u0663",
    "\u06F4": "\u0664", "\u06F5": "\u0665", "\u06F6": "\u0666", "\u06F7": "\u0667",
    "\u06F8": "\u0668", "\u06F9": "\u0669",
}

# Zero-width characters to remove
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
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize various whitespace characters."""
        # Replace multiple spaces with single space
        text = re.sub(r"[ \t]+", " ", text)
        # Normalize line endings
        text = re.sub(r"\r\n?", "\n", text)
        # Remove multiple blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text
    
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
