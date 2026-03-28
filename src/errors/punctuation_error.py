"""
Punctuation Error Generator

Sorani Kurdish uses a mix of Arabic-script punctuation (،  ؛  ؟) and
Latin punctuation (.  ,  ;  ?  !).  Common errors in learner and
informal text include:

  1. Deletion  — dropping a punctuation mark entirely
  2. Swap      — replacing Arabic comma ، with Latin comma , (or vice versa)
  3. Misplaced — placing a comma where a period should be, etc.
"""

import re
from typing import Optional

from .base import BaseErrorGenerator


# Punctuation swap pairs (original → possible replacements)
_PUNCT_SWAPS: dict[str, list[str]] = {
    "،": [",", ""],       # Arabic comma → Latin comma or deletion
    ",": ["،", ""],       # Latin comma → Arabic comma or deletion
    "؛": [";", "،"],      # Arabic semicolon → Latin semicolon or comma
    ";": ["؛", ","],
    "؟": ["?", ""],       # Arabic question mark → Latin or deletion
    "?": ["؟", ""],
    ".": ["،", ""],       # Period → comma or deletion
}


class PunctuationErrorGenerator(BaseErrorGenerator):
    """Generate punctuation errors (swap, delete, misplace)."""

    @property
    def error_type(self) -> str:
        return "punctuation"

    def find_eligible_positions(self, sentence: str) -> list[dict]:
        positions = []

        # Build pattern matching any target punctuation character
        punct_chars = re.escape("".join(_PUNCT_SWAPS.keys()))
        for match in re.finditer(rf'[{punct_chars}]', sentence):
            char = match.group()
            if char not in _PUNCT_SWAPS:
                continue
            replacement = self.rng.choice(_PUNCT_SWAPS[char])
            positions.append({
                "start": match.start(),
                "end": match.end(),
                "original": char,
                "context": {"replacement": replacement},
            })

        return positions

    def generate_error(self, position: dict) -> Optional[str]:
        return position["context"]["replacement"]
