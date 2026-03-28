"""
Word Order Error Generator

Sorani Kurdish is canonically SOV (Subject–Object–Verb), though pragmatic
fronting and topicalization allow some flexibility.  This generator
introduces word-order errors by:

  1. Moving the sentence-final verb to a non-final position (SOV → SVO)
  2. Swapping two adjacent non-verb constituents

These patterns reflect interference from SVO-dominant languages (English,
Arabic) commonly seen in learner writing.

Limitations:  The generator uses heuristic surface cues (verb prefixes,
sentence-final position) rather than a full parser.  It will miss
complex clauses and embedded structures.
"""

import re
from typing import Optional

from .base import BaseErrorGenerator
from ..data.tokenize import sorani_word_tokenize


# Common Sorani present-tense verb prefixes.
_VERB_PREFIXES = {"دە", "نا", "نە", "بـ", "ب"}


class WordOrderErrorGenerator(BaseErrorGenerator):
    """Generate SOV-violation errors by displacing the verb."""

    @property
    def error_type(self) -> str:
        return "word_order"

    def find_eligible_positions(self, sentence: str) -> list[dict]:
        words = sorani_word_tokenize(sentence)
        if len(words) < 3:
            return []

        positions = []

        # Build word offset map for precise span tracking (6A.8)
        word_offsets: list[tuple[int, int]] = [
            (m.start(), m.end()) for m in re.finditer(r'\S+', sentence)
        ]

        # Pattern 1: Move final verb to medial position (SOV → SVO)
        last_word = words[-1]
        is_verb = any(last_word.startswith(p) for p in _VERB_PREFIXES)
        if is_verb and len(words) >= 3:
            # Insert verb after subject (position 1)
            reordered = [words[0], last_word] + words[1:-1]
            # 6A.8: Full-sentence replacement is unavoidable for SOV→SVO,
            # but record which word indices actually moved for annotation.
            positions.append({
                "start": 0,
                "end": len(sentence),
                "original": sentence,
                "context": {
                    "pattern": "sov_to_svo",
                    "result": " ".join(reordered),
                    "moved_indices": [1, len(words) - 1],
                },
            })

        # Pattern 2: Swap two adjacent non-final words
        if len(words) >= 4:
            max_idx = len(words) - 2  # don't touch the last word
            if max_idx >= 1:
                idx = self.rng.randint(0, max_idx - 1)
                swapped = list(words)
                swapped[idx], swapped[idx + 1] = swapped[idx + 1], swapped[idx]
                # 6A.8: Replace only the two-word span, not the entire sentence.
                swap_start = word_offsets[idx][0]
                swap_end = word_offsets[idx + 1][1]
                local_result = words[idx + 1] + " " + words[idx]
                if local_result != sentence[swap_start:swap_end]:
                    positions.append({
                        "start": swap_start,
                        "end": swap_end,
                        "original": sentence[swap_start:swap_end],
                        "context": {
                            "pattern": "adjacent_swap",
                            "result": local_result,
                        },
                    })

        return positions

    def generate_error(self, position: dict) -> Optional[str]:
        return position["context"]["result"]
