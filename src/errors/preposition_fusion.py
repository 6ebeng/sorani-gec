"""
Preposition-Clitic Fusion Error Generator

Injects errors by "un-fusing" preposition-clitic fused forms, or by
applying the fused form incorrectly. Sorani Kurdish has three base
prepositions that fuse completely with pronominal clitics [F#33]:

    بە + clitic → پێ + clitic   (e.g. پێم = 'with me')
    لە + clitic → لێ + clitic   (e.g. لێت = 'from you')
    تە + clitic → تێ + clitic   (e.g. تێی = 'through it')

A common learner error is writing the un-fused analytic form instead
of the required synthetic fused form:
    *بە من  instead of  پێم
    *لە تۆ  instead of  لێت

The reverse error also occurs: applying fusion to prepositions that
should NOT fuse (e.g. *پێ after a non-pronoun complement).
"""

import re
from typing import Optional

from .base import BaseErrorGenerator
from ..morphology.constants import PP_INSEPARABLE


# Fused preposition-clitic paradigms [F#33]
# Each entry: (fused_form, analytic_preposition, independent_pronoun)
FUSION_PARADIGM: list[tuple[str, str, str]] = [
    # بە → پێ
    ("پێم", "بە", "من"),
    ("پێت", "بە", "تۆ"),
    ("پێی", "بە", "ئەو"),
    ("پێمان", "بە", "ئێمە"),
    ("پێتان", "بە", "ئیوە"),
    ("پێیان", "بە", "ئەوان"),
    # لە → لێ
    ("لێم", "لە", "من"),
    ("لێت", "لە", "تۆ"),
    ("لێی", "لە", "ئەو"),
    ("لێمان", "لە", "ئێمە"),
    ("لێتان", "لە", "ئیوە"),
    ("لێیان", "لە", "ئەوان"),
    # تە → تێ
    ("تێم", "تە", "من"),
    ("تێت", "تە", "تۆ"),
    ("تێی", "تە", "ئەو"),
    ("تێمان", "تە", "ئێمە"),
    ("تێتان", "تە", "ئیوە"),
    ("تێیان", "تە", "ئەوان"),
]

# Build lookup: fused_form → (preposition, pronoun)
_FUSED_TO_ANALYTIC: dict[str, tuple[str, str]] = {
    fused: (prep, pron) for fused, prep, pron in FUSION_PARADIGM
}

# All fused forms as a set for quick membership check
_FUSED_FORMS = frozenset(_FUSED_TO_ANALYTIC.keys())

# Build regex alternation for all fused forms (longest first to avoid
# partial matches)
_sorted_fused = sorted(_FUSED_FORMS, key=len, reverse=True)
_FUSED_PATTERN = re.compile(
    r'(?:^|(?<=\s))(' + '|'.join(re.escape(f) for f in _sorted_fused) + r')(?=\s|$)'
)


class PrepositionFusionErrorGenerator(BaseErrorGenerator):
    """Generate errors by splitting fused preposition-clitic forms.

    Correct:  پێم دەخوشە  ('she likes me')
    Error:    *بە من دەخوشە  (un-fused; analytic form is ungrammatical here)
    """

    @property
    def error_type(self) -> str:
        return "preposition_fusion"

    def find_eligible_positions(self, sentence: str) -> list[dict]:
        positions = []
        for match in _FUSED_PATTERN.finditer(sentence):
            fused = match.group(1)
            prep, pron = _FUSED_TO_ANALYTIC[fused]
            positions.append({
                "start": match.start(),
                "end": match.end(),
                "original": fused,
                "context": {
                    "preposition": prep,
                    "pronoun": pron,
                },
            })
        return positions

    def generate_error(self, position: dict) -> Optional[str]:
        ctx = position["context"]
        # Un-fuse: replace پێم with بە من
        return ctx["preposition"] + " " + ctx["pronoun"]
