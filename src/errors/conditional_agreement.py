"""
Conditional Agreement Error Generator

Generates errors by violating the tense-pairing rules in conditional sentences.

Targets:
  - Finding #103 (Conditional Sentence: 6 Tense-Pairing Laws)
  - Finding #149 (Conditional بـ as Sole Root–Clitic Intervenor)
  - Finding #215 (Past Conditional Clitic Ordering: Transitive/Intransitive Inversion)
"""

import re
from typing import Optional
from .base import BaseErrorGenerator


# ---------------------------------------------------------------------------
# Conditional markers
# ---------------------------------------------------------------------------
CONDITIONAL_MARKERS = ["ئەگەر", "ئەگەری", "ئەگەرنا", "هەر"]

# ---------------------------------------------------------------------------
# Tense morpheme patterns
# ---------------------------------------------------------------------------
# Present-tense prefixes
PRESENT_PREFIXES = ["دە", "ئە"]

# Conditional/subjunctive prefix
SUBJUNCTIVE_PREFIX = "ب"

# Past tense markers (recognisable stems)
PAST_TENSE_MARKERS = [
    "کرد", "چوو", "هات", "گوت", "دیت", "نووسی", "خوێند",
    "کڕی", "نیشت", "گەیشت", "بوو", "مرد", "کەوت",
    "خوارد", "دا", "ناردی", "بڕی", "فێربوو", "زانی",
    "وەستا", "ڕۆیشت", "گرت", "کوشت", "فرۆشت", "سووتا",
]

# Negation prefixes on verbs
NEGATION_PREFIXES = ["نە", "نا", "مە"]

# Compound verb preverbs
COMPOUND_PREVERBS = ["وەر", "هەڵ", "لێ", "تێ", "دەر", "پێ"]


class ConditionalAgreementErrorGenerator(BaseErrorGenerator):
    """Generate errors by injecting wrong tense in conditional sentence clauses.

    Finding #103 documents six tense-pairing laws for Sorani conditionals.
    For example, a real conditional pairs (present protasis, present apodosis)
    or (subjunctive protasis, present/future apodosis). Mixing a past-tense
    verb into the protasis of a present conditional, or putting a present
    verb into a counterfactual past conditional, creates an agreement error.

    This generator detects conditional sentences (via ئەگەر) and, when a
    subjunctive or present-tense verb is found in the protasis, replaces it
    with a past-tense form (or vice versa) to break the tense pairing.
    """

    @property
    def error_type(self) -> str:
        return "conditional_agreement"

    def find_eligible_positions(self, sentence: str) -> list[dict]:
        positions = []

        # Sentence must contain a conditional marker
        cond_alt = "|".join(re.escape(m) for m in CONDITIONAL_MARKERS)
        cond_match = re.search(rf'\b({cond_alt})\b', sentence)
        if cond_match is None:
            return positions

        cond_end = cond_match.end()

        # Identify apodosis boundary: look for a comma or semicolon
        clause_boundary = re.search(r'[،,؛]\s*', sentence[cond_end:])
        protasis_end = (cond_end + clause_boundary.start()) if clause_boundary else len(sentence)
        protasis = sentence[cond_end:protasis_end]

        # Build alternation for preverbs
        preverb_alt = "|".join(re.escape(p) for p in COMPOUND_PREVERBS)
        neg_alt = "|".join(re.escape(n) for n in NEGATION_PREFIXES)
        pres_alt = "|".join(re.escape(p) for p in PRESENT_PREFIXES)

        # --- Pattern A: subjunctive verb  بـ + stem + ending ---------------
        subj_pattern = re.compile(
            rf'\b((?:{neg_alt})?(?:{preverb_alt})?)(ب)(\w+?)(م|یت|ێت|ێ|ین|ن|ە|ات)\b'
        )
        for match in subj_pattern.finditer(protasis):
            full_start = cond_end + match.start()
            full_end = cond_end + match.end()
            positions.append({
                "start": full_start,
                "end": full_end,
                "original": match.group(),
                "context": {
                    "prefix": match.group(1),
                    "mood_marker": match.group(2),
                    "stem": match.group(3),
                    "ending": match.group(4),
                    "verb_tense": "subjunctive",
                },
            })

        # --- Pattern B: present verb  دە/ئە + stem + ending ----------------
        pres_pattern = re.compile(
            rf'\b((?:{neg_alt})?(?:{preverb_alt})?)((?:{pres_alt}))(\w+?)(م|یت|ێت|ێ|ین|ن|ات)\b'
        )
        for match in pres_pattern.finditer(protasis):
            full_start = cond_end + match.start()
            full_end = cond_end + match.end()
            # Avoid duplicating positions already found
            overlap = any(p["start"] <= full_start < p["end"] for p in positions)
            if not overlap:
                positions.append({
                    "start": full_start,
                    "end": full_end,
                    "original": match.group(),
                    "context": {
                        "prefix": match.group(1),
                        "mood_marker": match.group(2),
                        "stem": match.group(3),
                        "ending": match.group(4),
                        "verb_tense": "present",
                    },
                })

        return positions

    def generate_error(self, position: dict) -> Optional[str]:
        ctx = position["context"]
        stem = ctx["stem"]

        if ctx["verb_tense"] == "subjunctive":
            # Replace subjunctive بـ with present دەـ to break conditional pairing
            return ctx["prefix"] + "دە" + stem + ctx["ending"]
        else:
            # Present → subjunctive: replace دە/ئە with بـ
            return ctx["prefix"] + "ب" + stem + ctx["ending"]
