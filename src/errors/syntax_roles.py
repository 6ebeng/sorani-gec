"""
Syntax Role & Case Preposition Error Generator

Generates errors related to the misapplication of deep case roles and their
corresponding surface prepositions as described in Sa'id (2009).
Specifically targets Findings #137 to #140:
  - Agent (کارا): لەلایەن / لە لایەن
  - Instrument (ئامێر): بە
  - Experiencer (چێژەر): بۆ / بە

Learners often swap these due to L1 interference (e.g., English "by" mapping to 
both instrument "بە" and agent "لەلایەن" depending on context).
"""

import re
from typing import Optional
from .base import BaseErrorGenerator

class CaseRoleErrorGenerator(BaseErrorGenerator):
    """Generate errors by incorrectly swapping case role prepositions.
    
    Simulates learners confusing Agent, Instrument, and Benefactive/Experiencer 
    prepositions (e.g., using لەلایەن for instruments, or بە for passive agents).
    """
    
    @property
    def error_type(self) -> str:
        return "case_role_preposition"
        
    def find_eligible_positions(self, sentence: str) -> list[dict]:
        positions = []
        
        # 1. Agent markers (Finding #137 - Passive Agent)
        # Note: Often "لەلایەن" is paired with the enclitic "ەوە" later in the phrase,
        # but replacing the proposition itself is sufficient for a syntax error.
        for match in re.finditer(r'\b(لەلایەن|لە لایەن)\b', sentence):
            positions.append({
                "start": match.start(),
                "end": match.end(),
                "original": match.group(),
                "context": {"role": "agent"}
            })
            
        # 2. Instrument & Experiencer markers (Finding #137)
        # Matching standalone "بە" and "بۆ".
        for match in re.finditer(r'\b(بە|بۆ)\b', sentence):
            # Avoid overlap with previously found tokens
            overlap = any(pos["start"] <= match.start() < pos["end"] for pos in positions)
            if not overlap:
                pos_role = "instrument_experiencer" if match.group() == "بە" else "benefactive_experiencer"
                positions.append({
                    "start": match.start(),
                    "end": match.end(),
                    "original": match.group(),
                    "context": {"role": pos_role}
                })
        
        return positions

    def generate_error(self, position: dict) -> Optional[str]:
        orig = position["original"]
        role = position["context"]["role"]
        
        if role == "agent":
            # Finding #137/#140: Incorrectly map Agent to Instrument or Benefactive
            return self.rng.choice(["بە", "بۆ"])
            
        elif role in ["instrument_experiencer", "benefactive_experiencer"]:
            # Finding #137: Incorrectly map Instrument/Benefactive to Agent marker,
            # or swap them with each other. 
            # Error mapping e.g., "کردەوە بە کلیل" -> "کردەوە لەلایەن کلیل"
            # which absurdly treats the key as an agent instead of instrument.
            if orig == "بە":
                return self.rng.choice(["لەلایەن", "بۆ"])
            else:  # orig == "بۆ"
                return self.rng.choice(["لەلایەن", "بە"])
            
        return None
