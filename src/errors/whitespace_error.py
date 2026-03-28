"""
Whitespace / Segmentation Error Generator

Sorani Kurdish — written in Arabic script — lacks mandatory word-boundary
indicators, leading to frequent segmentation errors in informal and
learner text:

  1. Word fusion   — removing the space between two adjacent tokens
  2. Word splitting — inserting a space inside a long word

These errors are pervasive in social-media Kurdish text and represent a
significant chunk of real-world corrections that a GEC model must handle.
"""

import re
from typing import Optional

from .base import BaseErrorGenerator
from ..data.tokenize import sorani_word_tokenize


# Minimum word length before we attempt to split it.
_MIN_SPLIT_LEN = 5


class WhitespaceErrorGenerator(BaseErrorGenerator):
    """Generate word-boundary errors (fusion and splitting)."""

    @property
    def error_type(self) -> str:
        return "whitespace"

    def find_eligible_positions(self, sentence: str) -> list[dict]:
        words = sorani_word_tokenize(sentence)
        positions = []

        # --- Pattern 1: Fuse two adjacent words ---
        # 6B.2: Use word offsets directly instead of re.search(), which
        # only finds the first occurrence of a word pair and fails when
        # the same words appear multiple times in the sentence.
        word_offsets: list[tuple[int, int]] = [
            (m.start(), m.end()) for m in re.finditer(r'\S+', sentence)
        ]
        if len(words) >= 2:
            for i in range(len(words) - 1):
                w1 = words[i]
                w2 = words[i + 1]
                # Only fuse short tokens to keep the result plausible
                if len(w1) + len(w2) > 12:
                    continue
                fused = w1 + w2
                span_start = word_offsets[i][0]
                span_end = word_offsets[i + 1][1]
                positions.append({
                    "start": span_start,
                    "end": span_end,
                    "original": sentence[span_start:span_end],
                    "context": {
                        "pattern": "fusion",
                        "result": fused,
                    },
                })

        # --- Pattern 2: Split a long word ---
        for match in re.finditer(r'\S+', sentence):
            word = match.group()
            if len(word) < _MIN_SPLIT_LEN:
                continue
            # Split at a random interior position (at least 2 chars on each side)
            split_pos = self.rng.randint(2, len(word) - 2)
            split_result = word[:split_pos] + " " + word[split_pos:]
            positions.append({
                "start": match.start(),
                "end": match.end(),
                "original": word,
                "context": {
                    "pattern": "split",
                    "result": split_result,
                },
            })

        return positions

    def generate_error(self, position: dict) -> Optional[str]:
        return position["context"]["result"]
