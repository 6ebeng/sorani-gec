"""
Orthographic Error Generator

Simulates common spelling and orthographic errors as identified in the literature.

Targets:
  - Finding #163 (Allophonic Orthographic Pairs: ح↔ع↔هـ and خ↔غ)
  - Finding #164 (و/وو Short-Long Vowel Orthographic Confusion)
"""

import re
from typing import Optional
from .base import BaseErrorGenerator


class OrthographicErrorGenerator(BaseErrorGenerator):
    """Generate errors by swapping common orthographically confused characters.
    
    Native speakers frequently substitute Arabic loanword phones inconsistently 
    or under/over-double vowels like و.
    """

    @property
    def error_type(self) -> str:
        return "orthography"

    def find_eligible_positions(self, sentence: str) -> list[dict]:
        positions = []
        
        # Word boundary pattern
        for match in re.finditer(r'\b[^\s]+\b', sentence):
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
                # Deterministically pick the first option for reproducibility
                # Error Pipeline's error_rate will dictate if this actually gets applied.
                swap_to = sorted(options)[0]
                
                positions.append({
                    "start": match.start(),
                    "end": match.end(),
                    "original": word,
                    "context": {"swap_to": swap_to}
                })

        return positions

    def generate_error(self, position: dict) -> Optional[str]:
        return position["context"]["swap_to"]
