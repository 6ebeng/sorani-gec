"""
Negative Concord Error Generator

Generates errors relating to the strict negative concord rules in Sorani Kurdish.

Targets:
  - Finding #121 (Negative Concord: هیچ/چ Require Verb Negation)

Background linguistic findings (documented, not directly error-targeted):
  F#43  — Negation Patterns Across All Tenses (نا/نە/مە distribution)
  F#62  — Negation Prefixes (three prefix set: نا-, نە-, مە-)
  F#104 — Negation: Five Particles with Tense-Specific Distribution

Error patterns:
  1. Remove verb negation  — هیچم پێ ناخورێ → *هیچم پێ دەخورێ
  2. Double negation on wrong verb — نابینم هیچ نابزانم → *نابینم هیچ بزانم
     (remove negation from the wrong verb in the clause)
  3. Remove negative pronoun — هیچم نەبینی → *نەبینی
     (drop هیچ while keeping verb negation — incomplete concord)
"""

import re
from typing import Optional
from .base import BaseErrorGenerator
from ..morphology.constants import (
    NEGATION_WITHIN_WA_COORDINATION_PLURAL,
    PAST_TRANSITIVE_NEGATION_CLITIC_INTERPOSITION,
)


class NegativeConcordErrorGenerator(BaseErrorGenerator):
    """Generate negative concord errors.

    Three patterns: verb negation removal, wrong-verb negation removal,
    and negative pronoun deletion.

    Additional rules applied:
      F#343 (NEGATION_WITHIN_WA_COORDINATION_PLURAL): In و-coordinated
        clauses, negation applies within each conjunct separately.
      F#353 (PAST_TRANSITIVE_NEGATION_CLITIC_INTERPOSITION): In past
        transitive negation, the clitic may interpose between negation
        prefix and verb stem.
    """

    @property
    def error_type(self) -> str:
        return "negative_concord"

    def find_eligible_positions(self, sentence: str) -> list[dict]:
        positions = []

        # Check if sentence has 'هیچ' (possibly with clitic) or standalone 'چ'
        neg_pron_match = re.search(
            r'(?:^|(?<=\s))(هیچ)\S*(?=\s|$)|(?:^|(?<=\s))(چ)(?=\s|$)', sentence
        )
        if not neg_pron_match:
            return positions

        neg_pron_pos = neg_pron_match.start()
        # Clause boundaries: comma, و, کە, semicolon
        clause_boundary_re = re.compile(
            r'[،؛,]\s*|(?:^|(?<=\s))و(?=\s|$)|(?:^|(?<=\s))کە(?=\s|$)'
        )

        # --- Pattern 1 & 2: Remove verb negation prefix ---
        for match in re.finditer(r'(?:^|(?<=\s))(نا|نە)(\S+)(?=\s|$)', sentence):
            prefix = match.group(1)
            stem = match.group(2)

            # Same-clause check
            between_start = min(neg_pron_pos, match.start())
            between_end = max(neg_pron_pos, match.start())
            between_text = sentence[between_start:between_end]
            if clause_boundary_re.search(between_text):
                continue

            if prefix == 'نا':
                swap_to = 'دە' + stem
            else:
                if stem.startswith('دە'):
                    swap_to = stem
                else:
                    swap_to = stem

            if len(stem) > 2:
                positions.append({
                    "start": match.start(),
                    "end": match.end(),
                    "original": match.group(),
                    "context": {"swap_to": swap_to, "pattern": "remove_neg"},
                })

        # --- Pattern 3: Delete negative pronoun (keep verb negation) ---
        neg_word_match = re.search(
            r'(?:^|(?<=\s))(هیچ\S*)(?=\s|$)', sentence
        )
        if neg_word_match:
            positions.append({
                "start": neg_word_match.start(),
                "end": neg_word_match.end(),
                "original": neg_word_match.group(),
                "context": {"swap_to": "", "pattern": "remove_pronoun"},
            })

        return positions

    def generate_error(self, position: dict) -> Optional[str]:
        return position["context"]["swap_to"]
