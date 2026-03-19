"""
Dialectal Variation Error Generator

Generates morphological errors based on dialectal influences explicitly
detailed in the literature review, particularly from Northern and Southern
Kurdish dialects interfering with Standard Central Sorani writing.

Targets:
  - Finding #155 (Dialectal Participle Variation: و vs. ی vs. گ)
"""

import re
from typing import Optional
from .base import BaseErrorGenerator


class DialectalParticipleErrorGenerator(BaseErrorGenerator):
    """Generate errors by swapping the Standard Sorani participle morpheme 'و'
    with Northern ('ی') or Southern ('گ') dialect variants (Finding #155).
    
    Standard: هاتوە (hat-w-a), سوتاندوە (sutand-w-a), مردوە (mrd-w-a)
    Error (North): هاتیە (hat-y-a), سوتاندیە (sutand-y-a), مردیە (mrd-y-a)
    Error (South): هاتگە (hat-g-a), سوتاندگە (sutand-g-a), مردگە (mrd-g-a)
    """
    
    @property
    def error_type(self) -> str:
        return "dialectal_participle"
        
    def find_eligible_positions(self, sentence: str) -> list[dict]:
        positions = []
        
        # Match past participle + 'ە' (verb "to be" enclitic for 3sg perfect, or ezafe/def object).
        # We look for stems ending in 'و' before 'ە' or 'ەکە' etc., but to be safe 
        # and simple, we capture the common perfect tense ending "وە" and optionally 
        # personal endings like "وم", "ویت", "وە", "وین", "ون".
        # E.g., هاتوم (hat-w-m), هاتوە (hat-w-a).
        
        # Standard Sorani participle marker is 'و' following a consonant,
        # or 'وو' following a vowel. However, commonly in suffixes like
        # 'وە' (w-a), it's written as `وە` or `ووە`.
        
        # Pattern to match the participle 'و' followed by common clitics/endings:
        # e.g., وە (3sg), وم (1sg), ویت (2sg), وین (1pl), ون (2pl/3pl)
        # Note: We'll specifically target the 'وە' and 'ووە' endings to avoid false positives.
        
        # Capture 'و' or 'وو' followed by 'ە', 'ەم', 'ەت', 'مان', etc.
        # But specifically focusing on the morph "وە" at the end of a word or before clitics.
        pattern = r'([^\s]+)(ووە|وە)(?=\s|$)'
        
        for match in re.finditer(pattern, sentence):
            stem = match.group(1)
            suffix = match.group(2)
            
            # Prevent false positives with some common words
            if stem in ["ئە", "ئێ", "لێ", "پێ", "پیا"]:
                continue
                
            positions.append({
                "start": match.start(2),
                "end": match.end(2),
                "original": suffix,
                "context": {
                    "stem": stem,
                    "suffix": suffix
                }
            })
            
        return positions

    def generate_error(self, position: dict) -> Optional[str]:
        orig = position["original"]
        suffix = position["context"]["suffix"]
        
        # Finding #155 says 'و' is replaced by 'ی' (North) or 'گ' (South).
        # Standard: 'وە', 'ووە'
        # Targets: 'یە' / 'گە'
        
        if suffix == "وە":
            return self.rng.choice(["یە", "گە"])
        elif suffix == "ووە":
            # Just replacing the participle part
            return self.rng.choice(["یە", "گە"])
            
        return None
