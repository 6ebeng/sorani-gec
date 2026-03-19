"""
Quantifier-Verb Number Agreement Error Generator

Injects errors by flipping the verb from plural to singular when a
quantifier or numeral subject is present. In Sorani Kurdish, quantifiers
and numerals always force PLURAL agreement on the verb, regardless of the
noun's surface form [F#77, Slevanayi (2001), pp. 87-88]:

    دوو کەس هاتن    ('two people came')   → plural verb ✓
    *دوو کەس هات    ('two people came')   → singular verb ✗

The positional asymmetry matters [Mukriani (2000), pp. 24-26]: only
PRE-NOMINAL quantifiers (دوو کەس) control plural verb agreement.
Post-nominal ordinals (کەسی دووەم) do NOT — the noun controls instead.

This generator finds sentences with a pre-nominal quantifier followed
by a plural verb, and flips the verb to singular to create the error.
"""

import re
from typing import Optional

from .base import BaseErrorGenerator
from ..morphology.constants import QUANTIFIER_FORMS

# Present-tense plural verb endings that can be flipped to singular
# Source: Amin (2016), pp. 17-18
PLURAL_PRESENT_ENDINGS = {
    "ین": "م",     # 1pl → 1sg
    "ن": "ێت",     # 3pl → 3sg (most common with quantifier subjects)
    "ەن": "ێت",    # 3pl variant → 3sg
}

# Past-tense plural endings
# Source: Amin (2016), pp. 51-52; Slevanayi (2001), pp. 60-61
PLURAL_PAST_ENDINGS = {
    "ین": "م",       # 1pl → 1sg
    "ن": "",          # 3pl → 3sg (zero morpheme)
}

# Build regex for quantifier detection (longest first)
_sorted_quants = sorted(QUANTIFIER_FORMS, key=len, reverse=True)
_QUANTIFIER_PATTERN = re.compile(
    r'(?:^|(?<=\s))(' + '|'.join(re.escape(q) for q in _sorted_quants) + r')(?=\s)'
)


class QuantifierAgreementErrorGenerator(BaseErrorGenerator):
    """Generate errors by breaking quantifier → plural verb agreement.

    Correct:  زۆر کەس هاتن  ('many people came')
    Error:    *زۆر کەس هات  (singular verb with quantifier subject)
    """

    @property
    def error_type(self) -> str:
        return "quantifier_agreement"

    def find_eligible_positions(self, sentence: str) -> list[dict]:
        positions = []

        # Check if sentence contains a quantifier
        quant_match = _QUANTIFIER_PATTERN.search(sentence)
        if not quant_match:
            return positions

        # Find words after the quantifier (potential verb targets)
        words = sentence.split()
        quant_indices = []
        for i, w in enumerate(words):
            if w in QUANTIFIER_FORMS:
                quant_indices.append(i)

        if not quant_indices:
            return positions

        # Look for plural verbs in the sentence (after the quantifier)
        for qi in quant_indices:
            for vi in range(qi + 1, len(words)):
                word = words[vi]
                # Check present-tense plural endings
                for pl_end, sg_end in PLURAL_PRESENT_ENDINGS.items():
                    if word.endswith(pl_end) and len(word) > len(pl_end) + 1:
                        # Verify it looks like a verb (has a prefix or stem)
                        stem = word[:-len(pl_end)]
                        if self._looks_like_verb_stem(stem):
                            start = sentence.index(word, sum(len(words[j]) + 1 for j in range(vi)))
                            positions.append({
                                "start": start,
                                "end": start + len(word),
                                "original": word,
                                "context": {
                                    "stem": stem,
                                    "plural_ending": pl_end,
                                    "singular_ending": sg_end,
                                    "tense": "present",
                                },
                            })
                            break
                else:
                    # Check past-tense plural endings
                    for pl_end, sg_end in PLURAL_PAST_ENDINGS.items():
                        if word.endswith(pl_end) and len(word) > len(pl_end) + 1:
                            stem = word[:-len(pl_end)]
                            if self._looks_like_verb_stem(stem):
                                start = sentence.index(word, sum(len(words[j]) + 1 for j in range(vi)))
                                positions.append({
                                    "start": start,
                                    "end": start + len(word),
                                    "original": word,
                                    "context": {
                                        "stem": stem,
                                        "plural_ending": pl_end,
                                        "singular_ending": sg_end,
                                        "tense": "past",
                                    },
                                })
                                break

        return positions

    @staticmethod
    def _looks_like_verb_stem(stem: str) -> bool:
        """Heuristic: a verb stem has ≥2 chars and often starts with دە/بـ/نـ."""
        if len(stem) < 2:
            return False
        # Common verb prefixes
        if stem.startswith(("دە", "بـ", "نا", "نە", "مە", "بی", "ب")):
            return True
        # Past stems without prefix are also valid (≥3 chars)
        return len(stem) >= 3

    def generate_error(self, position: dict) -> Optional[str]:
        ctx = position["context"]
        # Replace plural ending with singular
        return ctx["stem"] + ctx["singular_ending"]
