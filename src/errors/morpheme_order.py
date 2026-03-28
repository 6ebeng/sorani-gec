"""
Morpheme Order Error Generator

Sorani Kurdish has a fixed internal ordering of derivational and inflectional
morphemes.  The canonical slot order for verbs is:

    NEG – MOOD – STEM – VOICE – TENSE/AGR – CLITIC

When learners or non-native writers violate this ordering — e.g., placing
the negation after the mood prefix, or attaching the clitic before the
tense suffix — the result is ungrammatical.

This generator simulates such errors by swapping adjacent morphological
components in common verb forms.  It operates at the surface level using
known prefix/suffix patterns rather than full morphological decomposition.
"""

import re
from typing import Optional

from .base import BaseErrorGenerator


# Negation + mood prefix combinations and their swapped versions
_PREFIX_SWAPS: dict[str, str] = {
    "نادە": "دەنا",     # neg+present → present+neg (wrong order)
    "نەدە": "دەنە",     # neg+present (نە variant)
    "نەب":  "بنە",      # neg+subjunctive → subj+neg
    "ناب":  "بنا",      # neg+subjunctive (نا variant)
}

# Common directional/completive verb particles and their misplaced forms
_PARTICLE_POSTVERB = {
    "دەچمەوە": "دەوەچم",      # 'go back' → misplaced 'وە'
    "دەگەڕێتەوە": "دەوەگەڕێت",
    "دەنێرێتەوە": "دەوەنێرێت",
}


class MorphemeOrderErrorGenerator(BaseErrorGenerator):
    """Generate morpheme-ordering errors inside verb forms."""

    @property
    def error_type(self) -> str:
        return "morpheme_order"

    def find_eligible_positions(self, sentence: str) -> list[dict]:
        positions = []

        for match in re.finditer(r'\S+', sentence):
            word = match.group()

            # Pattern 1: Prefix-order swap (NEG + MOOD)
            for correct, swapped in _PREFIX_SWAPS.items():
                if word.startswith(correct):
                    new_word = swapped + word[len(correct):]
                    positions.append({
                        "start": match.start(),
                        "end": match.end(),
                        "original": word,
                        "context": {"result": new_word,
                                    "pattern": "prefix_swap"},
                    })
                    break

            # Pattern 2: Directional particle misplacement
            if word in _PARTICLE_POSTVERB:
                positions.append({
                    "start": match.start(),
                    "end": match.end(),
                    "original": word,
                    "context": {"result": _PARTICLE_POSTVERB[word],
                                "pattern": "particle_misplace"},
                })

            # Pattern 3: Generic ەوە suffix misplacement — move ەوە
            # to after the first consonant cluster boundary, not the
            # fixed position 2. This produces more linguistically
            # plausible errors. (6A.6)
            if word.endswith("ەوە") and len(word) > 5:
                stem = word[:-3]  # strip ەوە
                # Find first consonant cluster boundary (after first
                # vowel following initial consonants). Kurdish vowels:
                _vowels = set("اەئێۆوویى")
                insert_pos = None
                found_vowel = False
                for ci, ch in enumerate(stem):
                    if ch in _vowels:
                        found_vowel = True
                    elif found_vowel:
                        # First consonant after first vowel = cluster boundary
                        insert_pos = ci
                        break
                if insert_pos is None or insert_pos < 2:
                    insert_pos = min(3, len(stem))  # fallback
                new_word = stem[:insert_pos] + "ەوە" + stem[insert_pos:]
                if new_word != word:
                    positions.append({
                        "start": match.start(),
                        "end": match.end(),
                        "original": word,
                        "context": {"result": new_word,
                                    "pattern": "suffix_misplace"},
                    })

        return positions

    def generate_error(self, position: dict) -> Optional[str]:
        return position["context"]["result"]
