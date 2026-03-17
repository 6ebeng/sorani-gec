"""
Base Error Generator

Abstract base class for all Sorani Kurdish agreement error generators.
Each generator targets a specific error type and can inject realistic
errors into clean Sorani text.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import random
import re


@dataclass
class ErrorAnnotation:
    """Annotation for a single injected error."""
    error_type: str          # e.g., "subject_verb_number"
    original_span: str       # original correct text
    error_span: str          # injected erroneous text
    start_pos: int           # character position in sentence
    end_pos: int             # character position in sentence
    description: str         # human-readable description


@dataclass
class ErrorResult:
    """Result of error injection on a sentence."""
    original: str                      # clean sentence
    corrupted: str                     # sentence with injected errors
    errors: list[ErrorAnnotation]      # list of injected errors
    
    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0
    
    def to_dict(self) -> dict:
        return {
            "original": self.original,
            "corrupted": self.corrupted,
            "errors": [
                {
                    "type": e.error_type,
                    "original": e.original_span,
                    "error": e.error_span,
                    "start": e.start_pos,
                    "end": e.end_pos,
                    "description": e.description,
                }
                for e in self.errors
            ],
        }


class BaseErrorGenerator(ABC):
    """Abstract base class for agreement error generators."""
    
    def __init__(self, error_rate: float = 0.15, seed: Optional[int] = None):
        """
        Args:
            error_rate: Probability of injecting an error at each eligible position.
            seed: Random seed for reproducibility.
        """
        self.error_rate = error_rate
        self.rng = random.Random(seed)
    
    @property
    @abstractmethod
    def error_type(self) -> str:
        """Return the error type identifier."""
        ...
    
    @abstractmethod
    def find_eligible_positions(self, sentence: str) -> list[dict]:
        """Find positions in the sentence where this error type can be injected.
        
        Returns:
            List of dicts with keys: 'start', 'end', 'original', 'context'
        """
        ...
    
    @abstractmethod
    def generate_error(self, position: dict) -> Optional[str]:
        """Generate an erroneous version of the text at the given position.
        
        Args:
            position: Dict from find_eligible_positions
            
        Returns:
            Erroneous text to replace the original, or None if no error can be generated.
        """
        ...
    
    def inject_errors(self, sentence: str) -> ErrorResult:
        """Inject errors into a clean sentence.
        
        Args:
            sentence: Clean Sorani Kurdish sentence.
            
        Returns:
            ErrorResult with original, corrupted text, and error annotations.
        """
        positions = self.find_eligible_positions(sentence)
        
        if not positions:
            return ErrorResult(original=sentence, corrupted=sentence, errors=[])
        
        # Decide which positions to corrupt
        selected = [
            pos for pos in positions
            if self.rng.random() < self.error_rate
        ]
        
        if not selected:
            return ErrorResult(original=sentence, corrupted=sentence, errors=[])
        
        # Sort by position (reverse) to apply replacements from end to start
        selected.sort(key=lambda p: p["start"], reverse=True)
        
        corrupted = sentence
        errors = []
        
        for pos in selected:
            error_text = self.generate_error(pos)
            if error_text is None:
                continue
            
            # Apply replacement
            corrupted = corrupted[:pos["start"]] + error_text + corrupted[pos["end"]:]
            # Clean up potential double spaces left from deletions 
            # (only if we replaced something with empty string or similar)
            corrupted = re.sub(r' +', ' ', corrupted).replace(' ,', ',').replace(' .', '.').strip()

            errors.append(ErrorAnnotation(
                error_type=self.error_type,
                original_span=pos["original"],
                error_span=error_text,
                start_pos=pos["start"],
                end_pos=pos["end"],
                description=f"{self.error_type}: '{pos['original']}' → '{error_text}'",
            ))
        
        return ErrorResult(original=sentence, corrupted=corrupted, errors=errors)
