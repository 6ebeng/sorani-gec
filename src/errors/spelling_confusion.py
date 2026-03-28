"""
Spelling Confusion Error Generator

Simulates character-level confusion errors common in Sorani Kurdish writing.
Distinct from orthography.py (which handles Arabic loanword phone swaps);
this generator targets confusion between graphemes that share visual
similarity or articulatory proximity specific to Kurdish:

  - ل ↔ ڵ  (lateral vs. dark lateral — F#163 extension)
  - ت ↔ ط  (Arabic emphatic / non-emphatic dental — loanwords)
  - د ↔ ض  (Arabic emphatic / non-emphatic voiced dental)
  - ر ↔ ڕ  (alveolar tap vs. trill; distinguished only in Kurdish script)
  - ح ↔ ھ  (pharyngeal vs. glottal — partially overlaps orthography.py)
  - ز ↔ ژ  (alveolar vs. postalveolar fricative)
"""

import re
from typing import Optional

from .base import BaseErrorGenerator


# Bidirectional confusion pairs with relative weight.
# Higher weight = more common confusion in learner/social-media text.
_CONFUSION_PAIRS: list[tuple[str, str, float]] = [
    ("ل", "ڵ", 3.0),   # very common — native distinction
    ("ر", "ڕ", 3.0),   # very common — native distinction
    ("ت", "ط", 1.5),   # moderately common — Arabic loanwords
    ("د", "ض", 1.0),   # less common — Arabic loanwords
    ("ز", "ژ", 1.5),   # moderately common
]


class SpellingConfusionErrorGenerator(BaseErrorGenerator):
    """Generate character-level confusion errors between visually or
    phonetically similar Kurdish graphemes."""

    @property
    def error_type(self) -> str:
        return "spelling_confusion"

    def find_eligible_positions(self, sentence: str) -> list[dict]:
        positions = []

        for match in re.finditer(r'\S+', sentence):
            word = match.group()
            swaps: list[tuple[str, float]] = []

            for char_a, char_b, weight in _CONFUSION_PAIRS:
                if char_a in word:
                    swaps.append((word.replace(char_a, char_b, 1), weight))
                if char_b in word:
                    swaps.append((word.replace(char_b, char_a, 1), weight))

            if not swaps:
                continue

            # Select one swap weighted by confusion frequency
            candidates, weights = zip(*swaps)
            chosen = self.rng.choices(list(candidates), weights=list(weights), k=1)[0]

            if chosen == word:
                continue

            positions.append({
                "start": match.start(),
                "end": match.end(),
                "original": word,
                "context": {"swap_to": chosen},
            })

        return positions

    def generate_error(self, position: dict) -> Optional[str]:
        return position["context"]["swap_to"]
