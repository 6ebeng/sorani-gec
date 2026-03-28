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
        # Core agent → patient pairs (Finding #159)
        "نووسەر": "نووسراو",       # writer → written
        "بکەر": "کراو",            # doer → done
        "کوژەر": "کوژراو",         # killer → killed
        "سووتێنەر": "سووتێنراو",   # burner → burnt
        "سووتێنەرە": "سوتاوە",     # burner (alt) → burnt
        "بیسەر": "بیستراو",        # listener → heard
        "بینەر": "بینراو",         # seer → seen
        "خوێنەر": "خوێندراو",     # reader → read
        "تێکدەر": "تێکدراو",      # mixer → mixed
        # Expanded pairs from Kurdish grammar resources
        "فرۆشەر": "فرۆشراو",      # seller → sold
        "کڕیار": "کڕدراو",        # buyer → bought
        "دروستکەر": "دروستکراو",  # maker → made
        "چاپکەر": "چاپکراو",     # printer → printed
        "داهێنەر": "داهێنراو",     # inventor → invented
        "دابەشکەر": "دابەشکراو",  # distributor → distributed
        "پەروەردەکەر": "پەروەردەکراو",  # educator → educated
        "ناردەر": "ناردراو",       # sender → sent
        "وەرگێڕەر": "وەرگێڕدراو", # translator → translated
        "ئامادەکەر": "ئامادەکراو", # preparer → prepared
        "بەرهەمهێنەر": "بەرهەمهێنراو",  # producer → produced
        "ڕێکخەر": "ڕێکخراو",      # organizer → organized
        "داواکەر": "داواکراو",     # requester → requested
        "پێشکەشکەر": "پێشکەشکراو", # presenter → presented
        "تاقیکەر": "تاقیکراو",    # tester → tested
        "ئاڵوگۆڕکەر": "ئاڵوگۆڕکراو", # exchanger → exchanged
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
        pattern = r'(?:^|(?<=\s))(' + '|'.join(self.ALL_WORDS) + r')(?=\s|$)'
        
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
