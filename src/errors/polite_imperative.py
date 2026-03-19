"""
Polite Imperative Error Generator

Generates errors by dropping or swapping obligatory preparatory phrases
(politeness markers) in polite imperative constructions.

In Sorani Kurdish, polite imperatives REQUIRE a preparatory phrase such as
تکایە ('please'), فەرموو/فەرمون ('go ahead'), بێ زەحمەت ('without trouble'),
ببوورە/ببوورن ('excuse me/us'), or بەڕێزتان ('with your respect').
Dropping the marker turns a polite request into a blunt command — a
register-level grammatical error common in learner text.

Targets:
  - Finding #162 (Polite Imperative Obligatory Preparatory Phrases)
"""

import re
from typing import Optional

from .base import BaseErrorGenerator

# ---------------------------------------------------------------------------
# Polite imperative markers (F#162)
# Source: constants.py — POLITE_IMPERATIVE_MARKERS
# ---------------------------------------------------------------------------
POLITE_MARKERS: tuple[str, ...] = (
    "فەرموو",       # go ahead (sg)
    "فەرمون",       # go ahead (pl)
    "تکایە",        # please
    "بێ زەحمەت",    # without trouble
    "بە یارمه‌تیت",  # with your help (ZWNJ inside)
    "ببوورن",       # excuse (pl)
    "ببوورە",       # excuse (sg)
    "بەڕێزتان",     # with your respect
)

# Sorted longest-first so multi-word markers match before single-word ones
_SORTED_MARKERS = sorted(POLITE_MARKERS, key=len, reverse=True)

# Build alternation for regex — escape each marker
_MARKER_ALT = "|".join(re.escape(m) for m in _SORTED_MARKERS)

# Pattern: a polite marker at or near the start of the sentence,
# optionally followed by a comma, then at least one more word.
_POLITE_PATTERN = re.compile(
    rf'(?:^|(?<=\s))({_MARKER_ALT})(?:\s*[،,])?\s+',
    re.UNICODE,
)

# Imperative verb prefix — helps confirm the sentence is imperative
_IMPERATIVE_PREFIX_RE = re.compile(
    r'(?:^|(?<=\s))(?:ب|مە)\w{2,}(?=\s|$)', re.UNICODE,
)

# Swap map: each marker → plausible wrong alternatives
# Used for error type 2 (swap marker rather than drop)
_MARKER_SWAPS: dict[str, list[str]] = {
    "فەرموو": ["فەرمون", "تکایە"],        # sg↔pl or register shift
    "فەرمون": ["فەرموو", "تکایە"],
    "تکایە":  ["فەرموو", "بێ زەحمەت"],
    "بێ زەحمەت": ["تکایە", "ببوورە"],
    "بە یارمه‌تیت": ["تکایە", "ببوورە"],
    "ببوورن": ["ببوورە", "فەرمون"],        # pl↔sg
    "ببوورە": ["ببوورن", "فەرموو"],
    "بەڕێزتان": ["تکایە", "ببوورن"],
}


class PoliteImperativeErrorGenerator(BaseErrorGenerator):
    """Generate errors by dropping or swapping polite imperative markers.

    Correct:  تکایە بنووسە  ('please write')
    Error:    *بنووسە        (bare command — missing politeness marker)
    Error:    *فەرموو بنووسە (wrong marker — register mismatch)
    """

    @property
    def error_type(self) -> str:
        return "polite_imperative"

    def find_eligible_positions(self, sentence: str) -> list[dict]:
        positions = []

        for match in _POLITE_PATTERN.finditer(sentence):
            marker = match.group(1)
            full_match = match.group(0)

            # Verify there is an imperative verb somewhere after the marker
            remainder = sentence[match.end():]
            if not _IMPERATIVE_PREFIX_RE.search(remainder):
                # Also accept the remainder as a plausible imperative
                # even without explicit prefix (some verbs are irregular)
                tokens_after = remainder.strip().split()
                if not tokens_after:
                    continue

            positions.append({
                "start": match.start(),
                "end": match.end(),
                "original": full_match,
                "context": {
                    "marker": marker,
                    "full_match": full_match,
                },
            })

        return positions

    def generate_error(self, position: dict) -> Optional[str]:
        ctx = position["context"]
        marker = ctx["marker"]

        # 70% chance: drop the marker entirely (more common error)
        # 30% chance: swap to a wrong marker
        if self.rng.random() < 0.7:
            # Drop — return empty string so the marker+trailing space vanish
            return ""
        else:
            alternatives = _MARKER_SWAPS.get(marker, [])
            if not alternatives:
                return ""
            new_marker = self.rng.choice(alternatives)
            # Replace the marker in the full match, preserving trailing space/comma
            return ctx["full_match"].replace(marker, new_marker, 1)
