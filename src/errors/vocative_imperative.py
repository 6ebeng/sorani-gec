"""
Vocative-Imperative Number Agreement Error Generator

Generates errors where vocative address and imperative verb disagree in number.

Targets:
  - Finding #76  (Imperative + Vocative Number Agreement)
  - Finding #202 (Vocative Case Triggers Full NP-Internal Adjective Agreement)
  - Finding #214 (Imperative ـە Epenthesis: Consonant-Final Singular vs Universal Plural ن)
  - Finding #226 (Free-Standing Vocative Particles with Gender Constraints)
"""

import re
from typing import Optional
from .base import BaseErrorGenerator


# ---------------------------------------------------------------------------
# Vocative markers (Finding #76, #147, #226)
# ---------------------------------------------------------------------------
# Singular vocative suffixes on nouns: ـۆ (masculine), ـێ (feminine/generic)
# Plural vocative: ـینۆ
VOCATIVE_SG_SUFFIXES = ["ۆ", "ێ"]
VOCATIVE_PL_SUFFIX = "ینۆ"

# Free-standing vocative particles (Finding #226): ئەی (masc.), گەلا (fem.)
VOCATIVE_PARTICLES = ["ئەی", "گەلا"]

# ---------------------------------------------------------------------------
# Imperative morphology (Finding #42, #214)
# ---------------------------------------------------------------------------
# Standard imperative prefix: بـ
# Prohibitive prefix: مەـ
# SG ending: ـە  (consonant-final stems) or ∅ (vowel-final stems)
# PL ending: ـن  (universal)
IMPERATIVE_SG_ENDING = "ە"
IMPERATIVE_PL_ENDING = "ن"

# Compound verb preverbs that appear before the imperative prefix
COMPOUND_PREVERBS = ["وەر", "هەڵ", "لێ", "تێ", "دەر", "پێ"]


class VocativeImperativeErrorGenerator(BaseErrorGenerator):
    """Generate errors by mismatching vocative address number with imperative verb number.

    In Sorani Kurdish, when a speaker addresses someone with a vocative noun
    (e.g., کوڕۆ 'hey boy-SG') the imperative verb must agree in number (بنووسە
    'write-SG!'). This generator flips the imperative ending so it disagrees
    with the vocative addressee: SG vocative + PL imperative, or vice versa.
    """

    @property
    def error_type(self) -> str:
        return "vocative_imperative"

    def find_eligible_positions(self, sentence: str) -> list[dict]:
        positions = []

        # Detect vocative number from the sentence
        vocative_number = self._detect_vocative_number(sentence)
        if vocative_number is None:
            return positions

        # Build alternation for preverbs
        preverb_alt = "|".join(re.escape(p) for p in COMPOUND_PREVERBS)

        # Match imperative verbs: (preverb?)(بـ|مەـ)(stem)(ە|ن)
        pattern = re.compile(
            rf'(?:^|(?<=\s))((?:{preverb_alt})?(?:ب|مە))(\w+?)(ە|ن)(?=\s|$)'
        )

        for match in pattern.finditer(sentence):
            prefix = match.group(1)
            stem = match.group(2)
            ending = match.group(3)

            # Skip very short stems (likely false positives)
            if len(stem) < 1:
                continue

            imp_number = "sg" if ending == IMPERATIVE_SG_ENDING else "pl"

            # Only flag if imperative number matches vocative number —
            # we will flip it to *create* a mismatch error
            if imp_number == vocative_number:
                positions.append({
                    "start": match.start(),
                    "end": match.end(),
                    "original": match.group(),
                    "context": {
                        "prefix": prefix,
                        "stem": stem,
                        "ending": ending,
                        "imp_number": imp_number,
                    },
                })

        return positions

    def generate_error(self, position: dict) -> Optional[str]:
        ctx = position["context"]
        # Flip the number: sg → pl and pl → sg
        if ctx["imp_number"] == "sg":
            new_ending = IMPERATIVE_PL_ENDING
        else:
            new_ending = IMPERATIVE_SG_ENDING
        return ctx["prefix"] + ctx["stem"] + new_ending

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_vocative_number(sentence: str) -> Optional[str]:
        """Return 'sg' or 'pl' if a vocative-marked noun is found, else None."""
        # Check plural vocative first (ـینۆ is longer, avoid false sg match)
        if re.search(r'[^\s]' + re.escape(VOCATIVE_PL_SUFFIX) + r'(?=\s|$)', sentence):
            return "pl"

        # Check singular vocative suffixes (ـۆ or ـێ after a stem)
        for suf in VOCATIVE_SG_SUFFIXES:
            if re.search(r'[^\s]' + re.escape(suf) + r'(?=\s|$)', sentence):
                return "sg"

        # Check free-standing vocative particles — default singular
        for particle in VOCATIVE_PARTICLES:
            if re.search(rf'(?:^|(?<=\s)){re.escape(particle)}(?=\s|$)', sentence):
                return "sg"

        return None
