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
                
            if options:
                # Filter to swaps that produce non-words (if lexicon available)
                if self._lexicon is not None:
                    invalid_options = [
                        o for o in options
                        if not self._lexicon.lookup(unicodedata.normalize("NFC", o))
                    ]
                    if invalid_options:
                        options = invalid_options
                    # If all options are valid words, keep them anyway (fall through)

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
