"""
Cross-Clause Agreement Error Generator

In Sorani Kurdish compound and complex sentences, the verbs in
coordinated clauses should maintain tense concordance when the temporal
frame is the same.  For example:

    ئەو هات و نیشتەوە      ('He came and sat down') — both past ✓
    *ئەو هات و دانیشێتەوە   ('He came and sits down') — tense clash ✗

This generator identifies coordinated clauses joined by conjunctions
(و، هەروەها، پاشان) and flips the tense of the second verb from
past to present or vice versa.
"""

import re
from typing import Optional

from .base import BaseErrorGenerator
from ..data.tokenize import sorani_word_tokenize


# Common coordinating conjunctions in Sorani Kurdish
_CONJUNCTIONS = {"و", "هەروەها", "پاشان", "بەڵام", "یان"}

# Simple past → present tense ending swaps (3sg as default)
# Fixed: 1sg "م" and 1pl "ین" had identity mappings (same output as input),
# meaning ~40% of swap attempts produced no actual tense change.
_PAST_TO_PRESENT: dict[str, str] = {
    "م": "ەم",         # 1sg: past -م → present -ەم (epenthetic vowel)
    "ی": "یت",         # 2sg
    "": "ێت",          # 3sg: zero ending → present
    "ین": "ەین",       # 1pl: past -ین → present -ەین
    "ن": "ەن",         # 3pl: past -ن → present -ەن
}

# Present-tense verb prefix
_PRESENT_PREFIX = "دە"


class CrossClauseAgreementErrorGenerator(BaseErrorGenerator):
    """Generate tense concordance errors between coordinated clauses."""

    @property
    def error_type(self) -> str:
        return "cross_clause_agreement"

    def find_eligible_positions(self, sentence: str) -> list[dict]:
        words = sorani_word_tokenize(sentence)
        positions = []

        # Find conjunction positions
        conj_indices = [
            i for i, w in enumerate(words) if w in _CONJUNCTIONS
        ]

        if not conj_indices:
            return positions

        # Build word offsets
        word_offsets: list[int] = [
            m.start() for m in re.finditer(r'\S+', sentence)
        ]

        for ci in conj_indices:
            # Look for a verb after the conjunction (within 3 words)
            for vi in range(ci + 1, min(ci + 4, len(words))):
                verb = words[vi]

                # 6A.2: Check negation prefix نە before checking present
                # prefix دە — negated past verbs (نەکرد) start with نە
                # and should NOT be classified as present tense.
                is_negated_past = verb.startswith("نە") and not verb.startswith("نەدە")

                # Past → present swap: if verb does NOT start with دە,
                # assume past tense and add the present prefix.
                if not verb.startswith(_PRESENT_PREFIX) and not is_negated_past and len(verb) >= 3:
                    new_verb = _PRESENT_PREFIX + verb
                    start = word_offsets[vi]
                    positions.append({
                        "start": start,
                        "end": start + len(verb),
                        "original": verb,
                        "context": {
                            "pattern": "past_to_present",
                            "result": new_verb,
                        },
                    })
                    break

                # Present → past swap: strip present prefix.
                # Also handle negated present (نادە → strip نا+دە)
                if verb.startswith("نا" + _PRESENT_PREFIX) and len(verb) > len("نا" + _PRESENT_PREFIX) + 1:
                    new_verb = "نە" + verb[len("نا" + _PRESENT_PREFIX):]
                    start = word_offsets[vi]
                    positions.append({
                        "start": start,
                        "end": start + len(verb),
                        "original": verb,
                        "context": {
                            "pattern": "neg_present_to_past",
                            "result": new_verb,
                        },
                    })
                    break

                if verb.startswith(_PRESENT_PREFIX) and len(verb) > len(_PRESENT_PREFIX) + 1:
                    new_verb = verb[len(_PRESENT_PREFIX):]
                    start = word_offsets[vi]
                    positions.append({
                        "start": start,
                        "end": start + len(verb),
                        "original": verb,
                        "context": {
                            "pattern": "present_to_past",
                            "result": new_verb,
                        },
                    })
                    break

        return positions

    def generate_error(self, position: dict) -> Optional[str]:
        return position["context"]["result"]
