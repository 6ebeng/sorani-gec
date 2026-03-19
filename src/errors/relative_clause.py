"""
Relative Clause Error Generator

Generates errors related to the relative clause marker 'کە' (that/which)
and its interaction with the ezafe marker 'ی' based on the findings from the literature.

Specifically targets:
  - Finding #141 (Relative Clause كە: Deletion, Ezafe, and Restrictiveness Rules)

Error strategies:
  A. Drop both 'کە' AND ezafe 'ی' from a definite relative clause head,
     producing an ungrammatical bare noun + clause sequence.
  B. Keep 'کە' but drop the obligatory ezafe 'ی' that links the head
     noun to the relative clause.
"""

import re
from typing import Optional
from .base import BaseErrorGenerator


class RelativeClauseErrorGenerator(BaseErrorGenerator):
    """Generate errors by mismanaging restrictive relative clause 'کە'.
    
    When 'کە' is present, the head noun must take the ezafe suffix 'ی' if 
    it is definite (ە). E.g., 'پەرداخەکەی کە من کڕیم'.
    If 'کە' is deleted (reduced relative clause), the ezafe 'ی' MUST remain.
    
    This generator produces two kinds of errors:
      A. Delete both 'کە' and 'ی' — learners wrongly drop everything.
      B. Keep 'کە' but delete 'ی' — the ezafe link is missing.
    """
    
    @property
    def error_type(self) -> str:
        return "relative_clause_ezafe"
        
    def find_eligible_positions(self, sentence: str) -> list[dict]:
        positions = []
        
        # Pattern 1: definite noun + ezafe + کە  (e.g. ەکەی کە, یەکەی کە)
        def_pattern = r'(\S+(?:ەکە|یەکە))ی\s+کە(?=\s|$)'
        for match in re.finditer(def_pattern, sentence):
            positions.append({
                "start": match.start(),
                "end": match.end(),
                "original": match.group(),
                "context": {
                    "head_noun": match.group(1),
                    "pattern": "definite",
                }
            })

        # Pattern 2: indefinite noun + ezafe + کە  (e.g. ێکی کە)
        indef_pattern = r'(\S+ێک)ی\s+کە(?=\s|$)'
        for match in re.finditer(indef_pattern, sentence):
            overlap = any(
                match.start() >= p["start"] and match.start() < p["end"]
                for p in positions
            )
            if overlap:
                continue
            positions.append({
                "start": match.start(),
                "end": match.end(),
                "original": match.group(),
                "context": {
                    "head_noun": match.group(1),
                    "pattern": "indefinite",
                }
            })

        # Pattern 3: bare/other noun + ezafe + کە
        # Match any word ending in ی followed by space+کە, excluding
        # matches already captured above.
        bare_pattern = r'(\S+)ی\s+کە(?=\s|$)'
        for match in re.finditer(bare_pattern, sentence):
            overlap = any(
                match.start() >= p["start"] and match.start() < p["end"]
                for p in positions
            )
            if overlap:
                continue
            # Require the stem to be at least 2 chars
            if len(match.group(1)) < 2:
                continue
            positions.append({
                "start": match.start(),
                "end": match.end(),
                "original": match.group(),
                "context": {
                    "head_noun": match.group(1),
                    "pattern": "bare",
                }
            })

        return positions

    def generate_error(self, position: dict) -> Optional[str]:
        ctx = position["context"]
        head = ctx["head_noun"]

        strategy = self.rng.random()
        if strategy < 0.5:
            # Strategy A: drop both 'ی' and 'کە'
            # e.g., "پەرداخەکەی کە" → "پەرداخەکە"
            return head
        else:
            # Strategy B: keep 'کە' but drop ezafe 'ی'
            # e.g., "پەرداخەکەی کە" → "پەرداخەکە کە"
            return head + " کە"
