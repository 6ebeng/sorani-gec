"""
Adversative Compound Sentence Error Generator

Generates errors related to the paired connector requirement in adversative
(contrastive) compound sentences.

Targets:
  - Finding #156 (Adversative Compound Sentence — Paired Connector Requirement)
"""

import re
from typing import Optional
from .base import BaseErrorGenerator, ErrorAnnotation, ErrorResult


class AdversativeConnectorErrorGenerator(BaseErrorGenerator):
    """Generate errors by deleting obligatory connectors in adversative sentences.
    
    According to Finding #156, a contrastive compound sentence requires at least
    one connector: an opening one (ئەگەرچی, هەرچەندە, لەگەڵ ئەوەشدا) OR a closing
    one (بەڵام, کەچی). 
    
    This generator simulates the error of completely dropping a single connector 
    when it's the only one bridging the two clauses, or dropping both if both 
    are present, creating an ungrammatical run-on sentence.
    """

    OPENERS = ["ئەگەرچی", "هەرچەندە", "لەگەڵ ئەوەشدا"]
    CLOSERS = ["بەڵام", "کەچی"]
    
    @property
    def error_type(self) -> str:
        return "missing_adversative_connector"
        
    def find_eligible_positions(self, sentence: str) -> list[dict]:
        positions = []
        
        opener_regex = r'(?:^|(?<=\s))(' + '|'.join(self.OPENERS) + r')(?=\s|$)'
        closer_regex = r'(?:^|(?<=\s))(' + '|'.join(self.CLOSERS) + r')(?=\s|$)'
        
        for match in re.finditer(opener_regex, sentence):
            positions.append({
                "start": match.start(),
                "end": match.end(),
                "original": match.group(),
                "context": {"type": "opener"}
            })
            
        for match in re.finditer(closer_regex, sentence):
            overlap = any(pos["start"] <= match.start() < pos["end"] for pos in positions)
            if not overlap:
                positions.append({
                    "start": match.start(),
                    "end": match.end(),
                    "original": match.group(),
                    "context": {"type": "closer"}
                })
        
        return positions

    def generate_error(self, position: dict) -> Optional[str]:
        """Generate error: either delete or substitute with wrong connector type.

        Pattern 1 (deletion): remove the connector entirely.
        Pattern 2 (substitution): replace opener with closer or vice versa,
            violating the paired-connector structure.
        """
        ctx = position["context"]

        # 60% deletion, 40% substitution
        if self.rng.random() < 0.6:
            return ""

        # Substitution: swap connector category
        if ctx["type"] == "opener":
            return self.rng.choice(self.CLOSERS)
        else:
            return self.rng.choice(self.OPENERS)

    def inject_errors(self, sentence: str, skip_word_indices=None) -> ErrorResult:
        """Override to handle both-connector sentences.

        When a sentence has both an opener AND a closer, the base class
        would only remove one (leaving the sentence grammatical).  This
        override removes ALL connectors in a single pass so the resulting
        sentence violates the paired-connector requirement.
        """
        positions = self.find_eligible_positions(sentence)
        if not positions:
            return ErrorResult(original=sentence, corrupted=sentence, errors=[])

        has_opener = any(p["context"]["type"] == "opener" for p in positions)
        has_closer = any(p["context"]["type"] == "closer" for p in positions)

        if has_opener and has_closer:
            # Must remove ALL connectors to create an error
            corrupted = sentence
            errors = []
            for pos in sorted(positions, key=lambda p: p["start"], reverse=True):
                error = self.generate_error(pos)
                if error is not None:
                    corrupted = corrupted[:pos["start"]] + error + corrupted[pos["end"]:]
                    errors.append(ErrorAnnotation(
                        error_type=self.error_type,
                        original_span=pos["original"],
                        error_span=error,
                        start_pos=pos["start"],
                        end_pos=pos["end"],
                        description=f"Deleted adversative connector '{pos['original']}'",
                    ))
            corrupted = re.sub(r' +', ' ', corrupted).strip()
            return ErrorResult(original=sentence, corrupted=corrupted, errors=errors)

        # Single connector — use base class behaviour (remove the one connector)
        return super().inject_errors(sentence)
