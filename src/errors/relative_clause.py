"""
Relative Clause Error Generator

Generates errors related to the relative clause marker 'کە' (that/which)
and its interaction with the ezafe marker 'ی' based on the findings from the literature.

Specifically targets:
  - Finding #141 (Relative Clause كە: Deletion, Ezafe, and Restrictiveness Rules)
"""

import re
from typing import Optional
from .base import BaseErrorGenerator


class RelativeClauseErrorGenerator(BaseErrorGenerator):
    """Generate errors by mismanaging restrictive relative clause 'کە'.
    
    When 'کە' is present, the head noun must take the ezafe suffix 'ی' if 
    it is definite (ە). E.g., 'پەرداخەکەی کە من کڕیم'.
    If 'کە' is deleted (reduced relative clause), the ezafe 'ی' MUST remain.
    
    This generator simulates the error of deleting 'کە' but ALSO deleting 
    the mandatory ezafe, creating an ungrammatical phrase: *'پەرداخەکە من کڕیم'.
    """
    
    @property
    def error_type(self) -> str:
        return "relative_clause_ezafe"
        
    def find_eligible_positions(self, sentence: str) -> list[dict]:
        positions = []
        
        # Look for definite noun + ezafe + کە (e.g. ەکەی کە)
        # We find "ەکەی کە" (aka -aka-y ka)
        # We will replace "ەکەی کە" with "ەکە" (erroneous deletion of both 'کە' and 'ی')
        
        pattern = r'(\S+ەکە)ی\s+کە\b'
        
        for match in re.finditer(pattern, sentence):
            positions.append({
                "start": match.start(),
                # end is the start + the length of "ەکەی کە", effectively the match end
                "end": match.end(),
                "original": match.group(),
                "context": {
                    "noun_def": match.group(1) # 'پەرداخەکە'
                }
            })
            
        return positions

    def generate_error(self, position: dict) -> Optional[str]:
        # Drop both ezafe ('ی') and 'کە'
        # e.g. "پەرداخەکەی کە" -> "پەرداخەکە"
        # The correct form, if reduced, would be "پەرداخەکەی", but learners might 
        # think removing 'کە' also removes any connector logic.
        return position["context"]["noun_def"]
