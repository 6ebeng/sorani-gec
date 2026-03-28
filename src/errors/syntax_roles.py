"""
Syntax Role & Case Preposition Error Generator

Generates errors related to the misapplication of deep case roles and their
corresponding surface prepositions as described in Sa'id (2009).
Specifically targets Findings #137 to #140:
  - Agent (کارا): لەلایەن / لە لایەن
  - Instrument (ئامێر): بە
  - Experiencer (چێژەر): بۆ / بە

Extended coverage also includes:
  - Location (شوێن): لە / لەسەر
  - Comitative (بەیەکەوە): لەگەڵ
  - Privative (بەدەر): بەبێ / بەبی

Learners often swap these due to L1 interference (e.g., English "by" mapping to
both instrument "بە" and agent "لەلایەن" depending on context).
"""

import re
from typing import Optional
from .base import BaseErrorGenerator


# Preposition → role mapping and swap targets
_PREPOSITION_SWAPS: dict[str, dict] = {
    "لەلایەن": {"role": "agent", "swaps": ["بە", "بۆ"]},
    "لە لایەن": {"role": "agent", "swaps": ["بە", "بۆ"]},
    "بە": {"role": "instrument", "swaps": ["لەلایەن", "بۆ", "لەگەڵ"]},
    "بۆ": {"role": "benefactive", "swaps": ["بە", "لە"]},
    "لە": {"role": "location", "swaps": ["لەسەر", "بە"]},
    "لەسەر": {"role": "surface_loc", "swaps": ["لە", "بە"]},
    "لەگەڵ": {"role": "comitative", "swaps": ["بە", "بۆ"]},
    "بەبێ": {"role": "privative", "swaps": ["بە", "بۆ"]},
    "بەبی": {"role": "privative", "swaps": ["بە", "بۆ"]},
}

class CaseRoleErrorGenerator(BaseErrorGenerator):
    """Generate errors by incorrectly swapping case role prepositions.

    Simulates learners confusing Agent, Instrument, Benefactive, Location,
    Comitative, and Privative prepositions.
    """

    @property
    def error_type(self) -> str:
        return "case_role_preposition"

    def find_eligible_positions(self, sentence: str) -> list[dict]:
        positions = []

        # Match all target prepositions; longer matches first to avoid
        # partial overlaps ('لەلایەن' before 'لە').
        sorted_preps = sorted(_PREPOSITION_SWAPS.keys(), key=len, reverse=True)
        prep_pattern = "|".join(re.escape(p) for p in sorted_preps)
        pattern = re.compile(
            rf'(?:^|(?<=\s))({prep_pattern})(?=\s|$)'
        )

        for match in pattern.finditer(sentence):
            word = match.group()
            if word not in _PREPOSITION_SWAPS:
                continue
            # Skip if overlapping with an already-found position
            if any(p["start"] <= match.start() < p["end"] for p in positions):
                continue
            info = _PREPOSITION_SWAPS[word]
            positions.append({
                "start": match.start(),
                "end": match.end(),
                "original": word,
                "context": {
                    "role": info["role"],
                    "swaps": info["swaps"],
                },
            })

        return positions

    def generate_error(self, position: dict) -> Optional[str]:
        return self.rng.choice(position["context"]["swaps"])
