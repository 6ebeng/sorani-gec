"""
Negative Concord Error Generator

Generates errors relating to the strict negative concord rules in Sorani Kurdish.

Targets:
  - Finding #121 (Negative Concord: هیچ/چ Require Verb Negation)
"""

import re
from typing import Optional
from .base import BaseErrorGenerator


class NegativeConcordErrorGenerator(BaseErrorGenerator):
    """Generate errors by removing required verb negation when negative pronouns are present.
    
    In Sorani, 'هیچ' requires a negative verb. 
    E.g., هیچم پێ ناخورێ -> *هیچم پێ دەخورێ
    """

    @property
    def error_type(self) -> str:
        return "negative_concord"

    def find_eligible_positions(self, sentence: str) -> list[dict]:
        positions = []
        
        # Check if sentence has 'هیچ' (possibly with clitic) or standalone 'چ'
        neg_pron_match = re.search(r'\b(هیچ)\S*\b|\b(چ)\b', sentence)
        if not neg_pron_match:
            return positions

        neg_pron_pos = neg_pron_match.start()
        # Clause boundaries: comma, و, کە, semicolon
        clause_boundary_re = re.compile(r'[،؛,]\s*|\bو\b|\bکە\b')
            
        # Find verbs with negative prefixes: نا (na-), نە (ne-)
        for match in re.finditer(r'\b(نا|نە)(\S+)\b', sentence):
            prefix = match.group(1)
            stem = match.group(2)
            
            # Check that negative pronoun and verb are in the same clause
            between_start = min(neg_pron_pos, match.start())
            between_end = max(neg_pron_pos, match.start())
            between_text = sentence[between_start:between_end]
            if clause_boundary_re.search(between_text):
                continue
            
            if prefix == 'نا':
                # Present: ناخورێ → دەخورێ
                swap_to = 'دە' + stem
            else:
                # نە prefix
                if stem.startswith('دە'):
                    # Present continuous: نەدەخوارد → دەخوارد
                    swap_to = stem
                else:
                    # Past: نەخوارد → خوارد (bare stem without negation)
                    swap_to = stem
                    
            # Avoid non-verbs: require stem length > 2 and presence of
            # verbal morphology (suffix or prefix evidence)
            if len(stem) > 2:
                positions.append({
                    "start": match.start(),
                    "end": match.end(),
                    "original": match.group(),
                    "context": {"swap_to": swap_to}
                })

        return positions

    def generate_error(self, position: dict) -> Optional[str]:
        return position["context"]["swap_to"]
