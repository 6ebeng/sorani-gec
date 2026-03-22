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

# Vowel-final present stems whose SG imperative has NO ـە epenthesis.
# Source: Amin (2016), pp. 23-24 — Finding #214
# These stems end in a vowel (ۆ, ێ, وو, ا) and take zero SG ending.
# E.g., بچۆ (go-SG!), بدۆ (run-SG!), بخوا (eat-SG!).
# PL always adds ن: بچن, بدن, بخون.
VOWEL_FINAL_STEMS = [
    "چۆ",       # go (present of چوون)
    "دۆ",       # run (present of دوان)
    "ڕۆ",       # go (dialectal, present of ڕۆیشتن)
    "خوا",      # eat (present of خواردن)
    "خۆ",       # eat (variant)
    "ژوا",      # chew (present of ژواردن)
    "وا",       # say (dialectal)
    "ڕا",       # run
    "کا",       # do (present of کردن)
    "با",       # carry (present of بردن)
    "دا",       # give (present of دان)
    "ها",       # come (present of هاتن)
]

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

        # --- Pattern A: consonant-final imperative verbs (بـ|مەـ)(stem)(ە|ن)
        pattern_a = re.compile(
            rf'(?:^|(?<=\s))((?:{preverb_alt})?(?:ب|مە))(\w+?)(ە|ن)(?=\s|$)'
        )

        for match in pattern_a.finditer(sentence):
            prefix = match.group(1)
            stem = match.group(2)
            ending = match.group(3)

            if len(stem) < 1:
                continue

            imp_number = "sg" if ending == IMPERATIVE_SG_ENDING else "pl"

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
                        "vowel_final": False,
                    },
                })

        # --- Pattern B: vowel-final SG imperatives (بـ|مەـ)(vowel-stem)
        # These have ZERO SG ending (no ـە epenthesis per Finding #214).
        vstem_alt = "|".join(re.escape(s) for s in VOWEL_FINAL_STEMS)
        pattern_b = re.compile(
            rf'(?:^|(?<=\s))((?:{preverb_alt})?(?:ب|مە))({vstem_alt})(?=\s|$)'
        )
        for match in pattern_b.finditer(sentence):
            prefix = match.group(1)
            stem = match.group(2)

            # Avoid overlap with Pattern A matches
            overlap = any(
                match.start() >= p["start"] and match.start() < p["end"]
                for p in positions
            )
            if overlap:
                continue

            # Vowel-final with no ending → singular
            imp_number = "sg"

            if imp_number == vocative_number:
                positions.append({
                    "start": match.start(),
                    "end": match.end(),
                    "original": match.group(),
                    "context": {
                        "prefix": prefix,
                        "stem": stem,
                        "ending": "",
                        "imp_number": imp_number,
                        "vowel_final": True,
                    },
                })

        return positions

    def generate_error(self, position: dict) -> Optional[str]:
        ctx = position["context"]
        # Flip the number: sg → pl and pl → sg
        if ctx["imp_number"] == "sg":
            new_ending = IMPERATIVE_PL_ENDING
        else:
            if ctx.get("vowel_final"):
                # pl → sg for vowel-final: drop ن, no ە
                new_ending = ""
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

        # Check singular vocative suffixes (ـۆ or ـێ after a stem of ≥2 chars)
        for suf in VOCATIVE_SG_SUFFIXES:
            if re.search(r'\S{2,}' + re.escape(suf) + r'(?=\s|$)', sentence):
                return "sg"

        # Check free-standing vocative particles — default singular
        for particle in VOCATIVE_PARTICLES:
            if re.search(rf'(?:^|(?<=\s)){re.escape(particle)}(?=\s|$)', sentence):
                return "sg"

        return None
