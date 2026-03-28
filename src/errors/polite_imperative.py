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

Background imperative-system findings (documented):
  F#61  — Imperative Markers (بـ positive, مە prohibition)
  F#359 — Subjunctive past three-marker ordering (ب > با > ایە)
  F#361 — Imperative -ە absorption with vowel-final roots
  F#366 — Imperative ب-marker omission in compound/preverbed verbs
  F#373 — Imperative -ڕە suffix (Sulaymaniyah emphatic singular)
"""

import re
from typing import Optional

from .base import BaseErrorGenerator
from ..data.tokenize import sorani_word_tokenize

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

# Imperative verb prefix — tightened to require stem + ending pattern
# instead of matching any word starting with ب/مە
_IMPERATIVE_PREFIX_RE = re.compile(
    r'(?:^|(?<=\s))(?:ب|مە)\w{2,}(?:ە|ن|ۆ|ێ)(?=\s|$)', re.UNICODE,
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

            # 6A.5: Validate imperative verb morphology after the marker.
            # The old code accepted any sentence with a polite marker even
            # when no imperative verb followed, causing false fires on
            # declarative sentences that happen to start with "تکایە".
            remainder = sentence[match.end():]
            has_imperative = _IMPERATIVE_PREFIX_RE.search(remainder)
            if not has_imperative:
                # Tighter fallback: require at least a 3+ char word that
                # could be an irregular imperative (e.g. "وەرە", "بچۆ").
                # Single-word remainders like punctuation are rejected.
                tokens_after = sorani_word_tokenize(remainder.strip())
                if not tokens_after or len(tokens_after[0]) < 3:
                    continue
                # Additional check: first word should not start with a
                # clear non-imperative prefix (present دە/ئە, past نە)
                first_token = tokens_after[0]
                if any(first_token.startswith(p) for p in ("دە", "ئە", "نە", "نا")):
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
