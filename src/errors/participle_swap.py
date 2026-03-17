"""
Participle Swap Error Generator

Generates errors relating to the confusion between agentive participles (ەر/ەرە)
and patient participles (او/راو), violating tense and voice constraints as 
identified in the literature.

Targets:
  - Finding #159 (Patient Participle (و) vs. Agent Participle (ر): Tense-Restricted Distribution)
"""

import re
from typing import Optional
from .base import BaseErrorGenerator


class ParticipleSwapErrorGenerator(BaseErrorGenerator):
    """Generate errors by incorrectly swapping Agent and Patient participles.
    
    Agent Participles (ەر) denote the doer (e.g., نووسەر - writer).
    Patient Participles (راو) denote the affected entity (e.g., نووسراو - written).
    Learners sometimes conflate these or use patient participles in present frames.
    """
    
    AGENT_TO_PATIENT = {
        "نووسەر": "نووسراو",
        "بکەر": "کراو",
        "کوژەر": "کوژراو",
        "سووتێنەر": "سووتێنراو",
        "سووتێنەرە": "سوتاوە",
        "بیسەر": "بیستراو",
        "بینەر": "بینراو",
        "خوێنەر": "خوێندراو",
        "تێکدەر": "تێکدراو"
    }
    
    # Reverse mapping
    PATIENT_TO_AGENT = {v: k for k, v in AGENT_TO_PATIENT.items()}
    
    # Combine keys for regex
    ALL_WORDS = list(AGENT_TO_PATIENT.keys()) + list(PATIENT_TO_AGENT.keys())
    
    @property
    def error_type(self) -> str:
        return "participle_voice_swap"
        
    def find_eligible_positions(self, sentence: str) -> list[dict]:
        positions = []
        
        # Build regex to match any of the words
        pattern = r'\b(' + '|'.join(self.ALL_WORDS) + r')\b'
        
        for match in re.finditer(pattern, sentence):
            # Check overlap
            overlap = any(pos["start"] <= match.start() < pos["end"] for pos in positions)
            if not overlap:
                word = match.group()
                swap_dest = self.AGENT_TO_PATIENT.get(word, self.PATIENT_TO_AGENT.get(word))
                
                positions.append({
                    "start": match.start(),
                    "end": match.end(),
                    "original": word,
                    "context": {"swap_to": swap_dest}
                })
                
        return positions

    def generate_error(self, position: dict) -> Optional[str]:
        return position["context"]["swap_to"]
