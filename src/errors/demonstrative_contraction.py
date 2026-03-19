"""
Demonstrative-Preposition Contraction Error Generator

Injects errors by splitting or failing to contract demonstrative+preposition
combinations. In Sorani Kurdish, بە/لە + ئەم/ئەو obligatorily contract
[F#123, Haji Marf (2014), pp. 263-264]:

    بە + ئەم → بەم       لە + ئەم → لەم
    بە + ئەو → بەو       لە + ئەو → لەو
    بە + ئەمە → بەمە     لە + ئەمە → لەمە
    بە + ئەوە → بەوە     لە + ئەوە → لەوە
    بە + ئەمانە → بەمانە لە + ئەمانە → لەمانە
    بە + ئەوانە → بەوانە لە + ئەوانە → لەوانە

Writing the uncontracted form (*بە ئەم instead of بەم) is a
segmentation/spelling error that learners produce frequently.
"""

import re
from typing import Optional

from .base import BaseErrorGenerator


# Contracted → (preposition, demonstrative) mapping [F#123]
CONTRACTIONS: dict[str, tuple[str, str]] = {
    "بەم": ("بە", "ئەم"),
    "لەم": ("لە", "ئەم"),
    "بەو": ("بە", "ئەو"),
    "لەو": ("لە", "ئەو"),
    "بەمە": ("بە", "ئەمە"),
    "لەمە": ("لە", "ئەمە"),
    "بەوە": ("بە", "ئەوە"),
    "لەوە": ("لە", "ئەوە"),
    "بەمانە": ("بە", "ئەمانە"),
    "لەمانە": ("لە", "ئەمانە"),
    "بەوانە": ("بە", "ئەوانە"),
    "لەوانە": ("لە", "ئەوانە"),
}

# Build regex: match contracted forms as whole words (longest first)
_sorted_contracted = sorted(CONTRACTIONS.keys(), key=len, reverse=True)
_CONTRACTION_PATTERN = re.compile(
    r'(?:^|(?<=\s))(' + '|'.join(re.escape(c) for c in _sorted_contracted) + r')(?=\s|$)'
)


class DemonstrativeContractionErrorGenerator(BaseErrorGenerator):
    """Generate errors by splitting obligatory demonstrative-preposition contractions.

    Correct:  لەم شارەدا
    Error:    *لە ئەم شارەدا  (uncontracted; incorrect segmentation)
    """

    @property
    def error_type(self) -> str:
        return "demonstrative_contraction"

    def find_eligible_positions(self, sentence: str) -> list[dict]:
        positions = []
        for match in _CONTRACTION_PATTERN.finditer(sentence):
            contracted = match.group(1)
            prep, dem = CONTRACTIONS[contracted]
            positions.append({
                "start": match.start(),
                "end": match.end(),
                "original": contracted,
                "context": {
                    "preposition": prep,
                    "demonstrative": dem,
                },
            })
        return positions

    def generate_error(self, position: dict) -> Optional[str]:
        ctx = position["context"]
        # Split: replace بەم with بە ئەم
        return ctx["preposition"] + " " + ctx["demonstrative"]
