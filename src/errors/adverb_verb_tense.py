"""
Adverb-Verb Tense Agreement Error Generator

Generates errors where a temporal adverb contradicts the tense of its verb.

Targets:
  - Finding #237 (Compound Sentence Verb Tense Concordance)
  - Finding #249 (Binary Tense System: Past vs. Non-Past Only)
  - Finding #256 (هەرگیز/هیچ Tense Restriction: Past and Future Only, Not Present)
"""

import re
from typing import Optional
from .base import BaseErrorGenerator


# ---------------------------------------------------------------------------
# Temporal adverbs mapped to their expected tense
# ---------------------------------------------------------------------------
PAST_ADVERBS = [
    "دوێنێ",        # yesterday
    "پێشتر",        # earlier / before
    "پار",          # last year
    "پاریەک",       # some time ago
    "پارێ",         # the other year
    "تازەگی",       # recently
    "لەمێژەوە",     # for a long time (past)
    "بەر لە ئێستا",  # before now
]

PRESENT_FUTURE_ADVERBS = [
    "ئێستا",        # now
    "سبەی",         # tomorrow
    "سبەینێ",       # tomorrow (variant)
    "داهاتوودا",     # in the future
    "ئەمڕۆ",        # today (present/future context)
    "هەمیشە",       # always
    "بەردەوام",      # continuously
]

# ---------------------------------------------------------------------------
# Verb morphology cues
# ---------------------------------------------------------------------------
# Present-tense prefixes
PRESENT_PREFIXES = ["دە", "ئە"]

# Past tense stems (subset; enough for high-coverage detection)
PAST_STEMS = [
    "کرد", "چوو", "هات", "گوت", "دیت", "نووسی", "خوێند",
    "کڕی", "نیشت", "گەیشت", "بوو", "مرد", "کەوت",
    "خوارد", "دا", "ناردی", "بڕی", "فێربوو", "زانی",
    "وەستا", "ڕۆیشت", "گرت", "کوشت", "فرۆشت", "سووتا",
]

# Negation prefixes
NEGATION_PREFIXES = ["نە", "نا"]

# Person/number clitics that can attach to past stems (Law 2 — ergative)
PAST_CLITICS = ["م", "ت", "ی", "مان", "تان", "یان"]

# Present endings (Law 1 — nominative)
PRESENT_ENDINGS = ["م", "یت", "ێت", "ێ", "ات", "ین", "ن", "ەم", "ەن"]

# Compound verb preverbs
COMPOUND_PREVERBS = ["وەر", "هەڵ", "لێ", "تێ", "دەر", "پێ"]


class AdverbVerbTenseErrorGenerator(BaseErrorGenerator):
    """Generate errors by swapping a verb's tense marker so it contradicts the temporal adverb.

    Example:
      correct:  دوێنێ چووم بۆ بازاڕ     (yesterday I-went to market)
      error:    دوێنێ دەچم بۆ بازاڕ     (yesterday I-go   to market)

    The generator finds sentences where a temporal adverb and verb tense agree,
    then flips the verb tense to create a mismatch.
    """

    @property
    def error_type(self) -> str:
        return "adverb_verb_tense"

    def find_eligible_positions(self, sentence: str) -> list[dict]:
        positions = []

        adverb_tense = self._detect_adverb_tense(sentence)
        if adverb_tense is None:
            return positions

        preverb_alt = "|".join(re.escape(p) for p in COMPOUND_PREVERBS)
        neg_alt = "|".join(re.escape(n) for n in NEGATION_PREFIXES)
        pres_alt = "|".join(re.escape(p) for p in PRESENT_PREFIXES)
        ending_alt = "|".join(re.escape(e) for e in PRESENT_ENDINGS)

        if adverb_tense == "past":
            # Adverb is past → look for past-tense verbs (we will flip them to present)
            for stem in PAST_STEMS:
                clitic_alt = "|".join(re.escape(c) for c in PAST_CLITICS)
                pattern = re.compile(
                    rf'(?:^|(?<=\s))((?:{neg_alt})?(?:{preverb_alt})?)'
                    rf'({re.escape(stem)})'
                    rf'({clitic_alt})?(?=\s|$)'
                )
                for match in pattern.finditer(sentence):
                    overlap = any(p["start"] <= match.start() < p["end"] for p in positions)
                    if overlap:
                        continue
                    # Skip if word has a present prefix (false positive)
                    word = match.group()
                    if any(word.startswith(pp) for pp in PRESENT_PREFIXES):
                        continue
                    positions.append({
                        "start": match.start(),
                        "end": match.end(),
                        "original": word,
                        "context": {
                            "prefix": match.group(1),
                            "stem": match.group(2),
                            "clitic": match.group(3) or "",
                            "verb_tense": "past",
                        },
                    })

        elif adverb_tense == "present":
            # Adverb is present/future → look for present-tense verbs (flip to past)
            pres_pattern = re.compile(
                rf'(?:^|(?<=\s))((?:{neg_alt})?(?:{preverb_alt})?)'
                rf'({pres_alt})'
                rf'(\w+?)'
                rf'({ending_alt})(?=\s|$)'
            )
            for match in pres_pattern.finditer(sentence):
                overlap = any(p["start"] <= match.start() < p["end"] for p in positions)
                if overlap:
                    continue
                positions.append({
                    "start": match.start(),
                    "end": match.end(),
                    "original": match.group(),
                    "context": {
                        "prefix": match.group(1),
                        "tense_marker": match.group(2),
                        "stem": match.group(3),
                        "ending": match.group(4),
                        "verb_tense": "present",
                    },
                })

        return positions

    def generate_error(self, position: dict) -> Optional[str]:
        ctx = position["context"]

        if ctx["verb_tense"] == "past":
            # Past → present: wrap the stem with دە...+ending
            # Map the past clitic to a present ending (approximate same PN)
            ending = self._past_clitic_to_present_ending(ctx["clitic"])
            return ctx["prefix"] + "دە" + ctx["stem"] + ending

        else:
            # Present → past: strip the present prefix, use stem + past clitic
            # Map the present ending back to a past-tense clitic to produce
            # a grammatical (but tense-mismatched) past form.
            past_clitic = self._present_ending_to_past_clitic(ctx["ending"])
            return ctx["prefix"] + ctx["stem"] + past_clitic

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _detect_adverb_tense(self, sentence: str) -> Optional[str]:
        """Return 'past' or 'present' based on the temporal adverb found."""
        for adv in PAST_ADVERBS:
            if re.search(rf'(?:^|(?<=\s)){re.escape(adv)}(?=\s|$)', sentence):
                return "past"
        for adv in PRESENT_FUTURE_ADVERBS:
            if re.search(rf'(?:^|(?<=\s)){re.escape(adv)}(?=\s|$)', sentence):
                return "present"
        return None

    @staticmethod
    def _past_clitic_to_present_ending(clitic: str) -> str:
        """Map a past-tense clitic to an approximate present-tense ending."""
        mapping = {
            "م": "م",      # 1sg
            "ت": "یت",     # 2sg
            "ی": "ێت",     # 3sg
            "مان": "ین",   # 1pl
            "تان": "ن",    # 2pl
            "یان": "ن",    # 3pl
            "": "ێت",      # bare 3sg default
        }
        return mapping.get(clitic, "ێت")

    @staticmethod
    def _present_ending_to_past_clitic(ending: str) -> str:
        """Map a present-tense ending to an approximate past-tense clitic."""
        mapping = {
            "م": "م",      # 1sg
            "ەم": "م",     # 1sg (epenthetic)
            "یت": "ت",     # 2sg
            "ێت": "",      # 3sg → zero morpheme
            "ات": "",      # 3sg (after -a stems)
            "ێ": "",       # 3sg (short form)
            "ین": "مان",   # 1pl
            "ن": "ن",      # 3pl
            "ەن": "ن",     # 3pl (epenthetic)
        }
        return mapping.get(ending, "")
