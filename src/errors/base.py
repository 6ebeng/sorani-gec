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
import unicodedata


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
    source_id: str | None = None       # source article/document identifier
    
    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0
    
    def to_dict(self) -> dict:
        d = {
            "original": self.original,
            "corrupted": self.corrupted,
            "source": self.corrupted,
            "target": self.original,
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
        if self.source_id is not None:
            d["source_id"] = self.source_id
        return d


class BaseErrorGenerator(ABC):
    """Abstract base class for agreement error generators."""
    
    def __init__(
        self,
        error_rate: float = 0.15,
        seed: Optional[int] = None,
        analyzer: Optional["MorphologicalAnalyzer"] = None,
    ):
        """
        Args:
            error_rate: Probability of injecting an error at each eligible position.
            seed: Random seed for reproducibility.
            analyzer: Optional MorphologicalAnalyzer for dynamic verb/noun
                recognition.  When provided, subclasses can call
                ``self.analyzer.analyze_token(word)`` instead of relying
                solely on hardcoded stem lists.
        """
        self.error_rate = error_rate
        self.rng = random.Random(seed)
        self.analyzer = analyzer
    
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
    
    def inject_errors(
        self,
        sentence: str,
        skip_word_indices: Optional[set[int]] = None,
    ) -> ErrorResult:
        """Inject errors into a clean sentence.
        
        Args:
            sentence: Clean Sorani Kurdish sentence.
            skip_word_indices: Word indices (0-based) to exclude from
                corruption.  Used by the pipeline to prevent a second
                generator from flipping a token that was already
                corrupted by a prior generator (double-flip avoidance).
            
        Returns:
            ErrorResult with original, corrupted text, and error annotations.
        """
        positions = self.find_eligible_positions(unicodedata.normalize("NFC", sentence))
        
        if not positions:
            return ErrorResult(original=sentence, corrupted=sentence, errors=[])
        
        # Filter out positions overlapping with previously corrupted words
        # 6B.3: Track char-level modified ranges alongside word indices.
        # Generators producing char-level replacements (e.g., mid-morpheme
        # swaps in orthography.py) can bypass the word-index guard.
        # This adds char-range overlap checking as a secondary filter.
        if skip_word_indices:
            word_ranges: list[tuple[int, int]] = [
                (m.start(), m.end())
                for m in re.finditer(r'\S+', sentence)
            ]

            # Build char-level modified ranges from skip_word_indices
            skip_char_ranges: list[tuple[int, int]] = []
            for wi in skip_word_indices:
                if wi < len(word_ranges):
                    skip_char_ranges.append(word_ranges[wi])

            def _overlaps_skip(pos: dict) -> bool:
                # Check word-level overlap (backwards-compatible)
                for wi in skip_word_indices:
                    if wi < len(word_ranges):
                        wr_s, wr_e = word_ranges[wi]
                        if pos["start"] < wr_e and pos["end"] > wr_s:
                            return True
                # Check char-level overlap for sub-word edits
                for cr_s, cr_e in skip_char_ranges:
                    if pos["start"] < cr_e and pos["end"] > cr_s:
                        return True
                return False

            positions = [p for p in positions if not _overlaps_skip(p)]

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
        # This ensures earlier positions' indices remain valid.
        selected.sort(key=lambda p: p["start"], reverse=True)
        
        corrupted = sentence
        errors = []
        
        for pos in selected:
            error_text = self.generate_error(pos)
            if error_text is None:
                continue
            
            # Apply replacement (right-to-left keeps upstream positions stable)
            corrupted = corrupted[:pos["start"]] + error_text + corrupted[pos["end"]:]

            # Record the error with positions in the ORIGINAL sentence
            errors.append(ErrorAnnotation(
                error_type=self.error_type,
                original_span=pos["original"],
                error_span=error_text,
                start_pos=pos["start"],
                end_pos=pos["end"],
                description=f"{self.error_type}: '{pos['original']}' → '{error_text}'",
            ))
        
        # Clean up whitespace ONCE after all replacements are applied
        corrupted = re.sub(r' +', ' ', corrupted).replace(' ,', ',').replace(' .', '.')
        corrupted = corrupted.replace(' ،', '،').replace(' ؛', '؛').strip()
        
        # Apply the same whitespace cleanup to each error_span so the
        # annotation text matches the final corrupted string.
        for err in errors:
            cleaned = re.sub(r' +', ' ', err.error_span).strip()
            if cleaned != err.error_span:
                err.error_span = cleaned
        
        return ErrorResult(original=sentence, corrupted=corrupted, errors=errors)
