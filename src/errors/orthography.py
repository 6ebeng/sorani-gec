"""
Orthographic Error Generator

Simulates common spelling and orthographic errors as identified in the literature.

Targets:
  - Finding #163 (Allophonic Orthographic Pairs: ح↔ع↔هـ and خ↔غ)
  - Finding #164 (و/وو Short-Long Vowel Orthographic Confusion)
"""

import re
import unicodedata
from typing import Optional
from .base import BaseErrorGenerator


class OrthographicErrorGenerator(BaseErrorGenerator):
    """Generate errors by swapping common orthographically confused characters.
    
    Native speakers frequently substitute Arabic loanword phones inconsistently 
    or under/over-double vowels like و.
    """

    def __init__(self, *args, lexicon=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._lexicon = lexicon

    @property
    def error_type(self) -> str:
        return "orthography"

    def find_eligible_positions(self, sentence: str) -> list[dict]:
        positions = []
        
        # Match non-whitespace sequences (Arabic script doesn't work with \b)
        for match in re.finditer(r'(?:^|(?<=\s))([^\s]+)(?=\s|$)', sentence):
            word = match.group()
            
            options = []
            
            # Finding #163 substitutions
            if 'ح' in word: 
                options.append(word.replace('ح', 'ه'))
            if 'ه' in word and not word.endswith('ە'): # Avoid the standard vowel 'ە'
                options.append(word.replace('ه', 'ح'))
            if 'غ' in word: 
                options.append(word.replace('غ', 'خ'))
            if 'خ' in word: 
                options.append(word.replace('خ', 'غ'))
                
            # Finding #164 substitutions
            if 'وو' in word: 
                options.append(word.replace('وو', 'و', 1)) 
            elif 'و' in word and 'وو' not in word:
                options.append(word.replace('و', 'وو', 1))

            # PIPE-22: Additional orthographic confusion pairs common
            # in Sorani Kurdish writing (native speaker errors).
            # ئ↔ی word-initially: initial glottal stop vs yeh
            if word.startswith('ئ'):
                options.append('ی' + word[1:])
            elif word.startswith('ی'):
                options.append('ئ' + word[1:])
            # ع↔ئ: Arabic ʕayn vs glottal stop (loanword confusion)
            if 'ع' in word:
                options.append(word.replace('ع', 'ئ', 1))
            if 'ئ' in word and not word.startswith('ئ'):
                options.append(word.replace('ئ', 'ع', 1))
            # ڵ↔ل: velarized lateral vs plain lateral
            if 'ڵ' in word:
                options.append(word.replace('ڵ', 'ل', 1))
            if 'ل' in word:
                options.append(word.replace('ل', 'ڵ', 1))
            # ڕ↔ر: trilled r vs tap r
            if 'ڕ' in word:
                options.append(word.replace('ڕ', 'ر', 1))
            if 'ر' in word:
                options.append(word.replace('ر', 'ڕ', 1))
            # ێ↔ی: long vowel vs short vowel
            if 'ێ' in word:
                options.append(word.replace('ێ', 'ی', 1))
            if 'ی' in word and not word.startswith('ی'):
                options.append(word.replace('ی', 'ێ', 1))
            # ۆ↔و: rounded mid vowel vs high vowel
            if 'ۆ' in word:
                options.append(word.replace('ۆ', 'و', 1))
            if 'و' in word and 'وو' not in word and 'ۆ' not in word:
                options.append(word.replace('و', 'ۆ', 1))
            # ژ↔ز: voiced postalveolar vs alveolar fricative
            if 'ژ' in word:
                options.append(word.replace('ژ', 'ز', 1))
            if 'ز' in word:
                options.append(word.replace('ز', 'ژ', 1))
            # ڤ↔ف: voiced vs voiceless labiodental fricative
            if 'ڤ' in word:
                options.append(word.replace('ڤ', 'ف', 1))
            if 'ف' in word:
                options.append(word.replace('ف', 'ڤ', 1))
                
            if options:
                # Filter to swaps that produce non-words (if lexicon available)
                if self._lexicon is not None:
                    invalid_options = [
                        o for o in options
                        if not self._lexicon.lookup(unicodedata.normalize("NFC", o))
                    ]
                    if invalid_options:
                        options = invalid_options
                    else:
                        # 6A.7: ALL swaps produce valid Sorani words.
                        # Injecting such a swap creates a false positive
                        # ("corrupted" sentence that is actually correct).
                        # Skip this word entirely.
                        continue

                swap_to = self.rng.choice(options)
                
                positions.append({
                    "start": match.start(),
                    "end": match.end(),
                    "original": word,
                    "context": {"swap_to": swap_to}
                })

        return positions

    def generate_error(self, position: dict) -> Optional[str]:
        return position["context"]["swap_to"]
