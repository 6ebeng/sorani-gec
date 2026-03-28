"""
Possessive Clitic Error Generator

Injects errors by confusing possessive clitics with agreement clitics.
In Sorani Kurdish, the six bound morphemes (م/ت/ی/مان/تان/یان) serve
both possessive and agreement functions, but their distribution is
strictly constrained [F#21, F#71]:

    Possessive clitics attach to NOUNS only:
        کتێبم  ('my book')  — possessive م on noun ✓
        *دەخوێنم  — possessive م on verb ✗ (this is agreement, not possession)

    Possessive pronouns NEVER trigger verb agreement [F#71]:
        کتێبم باشە  ('my book is good')
        The م on کتێب is possessive; the verb باش+ە agrees with
        کتێب (3sg), NOT with the possessor من (1sg).

    F#83: Two pronoun sets with opposite agreement behaviour:
        Independent pronouns (من, تۆ, ...) → DO trigger agreement
        Bound possessives (م, ت, ...) on nouns → do NOT trigger agreement

Common errors:
    1. Swapping the possessive clitic for a wrong person (same surface
       confusion as regular clitic errors, but on nominal hosts)
    2. Possessive clitic on verb stems (hypercorrection)

Source: Slevanayi (2001), pp. 77-78; Haji Marif (2014), Chapter 7
"""

import re
from typing import Optional

from .base import BaseErrorGenerator

# Possessive clitics (same morphemes as agreement clitics)
# Source: Slevanayi (2001), pp. 77-78 [F#71]
POSSESSIVE_CLITICS = {
    "م": "1sg",     # my
    "ت": "2sg",     # your (sg)
    "ی": "3sg",     # his/her
    "مان": "1pl",   # our
    "تان": "2pl",   # your (pl)
    "یان": "3pl",   # their
}

# Swap mapping: each possessive clitic → incorrect alternatives
# Prioritizes within-number swaps (more natural errors)
POSSESSIVE_SWAPS = {
    "م":   ["ت", "ی", "مان"],
    "ت":   ["م", "ی", "تان"],
    "ی":   ["م", "ت", "یان"],
    "مان": ["تان", "یان", "م"],
    "تان": ["مان", "یان", "ت"],
    "یان": ["مان", "تان", "ی"],
}

# Common noun suffixes that indicate the host is a noun (not a verb)
# Definite: ەکە (the), indefinite: ێک (a), demonstrative: ە (this/that)
# The possessive clitic attaches AFTER these suffixes:
#   کتێبەکەم ('my book'), کتێبێکم ('a book of mine')
NOUN_SUFFIXES = ["ەکە", "ێک", "ەک", "انە", "ان"]

# Common verb prefixes — if a word starts with these, it is likely a verb
# and should NOT host a possessive clitic in standard grammar.
VERB_PREFIXES = ("دە", "ب", "نا", "نە", "مە", "بی")

# Build regex for detecting noun+possessive patterns
# Match: (noun_stem)(optional_noun_suffix)(possessive_clitic) at word boundary
_sorted_clitics = sorted(POSSESSIVE_CLITICS.keys(), key=len, reverse=True)
_CLITIC_ALT = '|'.join(re.escape(c) for c in _sorted_clitics)
_NOUN_SUFFIX_ALT = '|'.join(re.escape(s) for s in sorted(NOUN_SUFFIXES, key=len, reverse=True))

# Pattern: word with an optional noun-like suffix followed by a possessive clitic.
# The suffix is optional to handle bare stem + clitic cases (e.g., کاتی 'his time').
_NOUN_POSSESSIVE_PATTERN = re.compile(
    rf'(?:^|(?<=\s))([\u0600-\u06FF]{{2,}}(?:{_NOUN_SUFFIX_ALT})?)'
    rf'({_CLITIC_ALT})(?=\s|$)'
)


class PossessiveCliticErrorGenerator(BaseErrorGenerator):
    """Generate errors by swapping possessive clitics on nouns.

    Correct:  کتێبەکەم  ('my book')
    Error:    *کتێبەکەت  ('your book' — wrong possessor)
    """

    @property
    def error_type(self) -> str:
        return "possessive_clitic"

    def find_eligible_positions(self, sentence: str) -> list[dict]:
        positions = []

        for match in _NOUN_POSSESSIVE_PATTERN.finditer(sentence):
            noun_part = match.group(1)
            clitic = match.group(2)

            # Skip if the stem looks like a verb (starts with verb prefix)
            if noun_part.startswith(VERB_PREFIXES):
                continue

            # Skip very short stems
            if len(noun_part) < 3:
                continue

            full_word = match.group(0)
            positions.append({
                "start": match.start(),
                "end": match.end(),
                "original": full_word,
                "context": {
                    "noun_part": noun_part,
                    "clitic": clitic,
                },
            })

        return positions

    def generate_error(self, position: dict) -> Optional[str]:
        ctx = position["context"]
        current = ctx["clitic"]

        if current not in POSSESSIVE_SWAPS:
            return None

        alternatives = POSSESSIVE_SWAPS[current]
        if not alternatives:
            return None

        new_clitic = self.rng.choice(alternatives)
        error_word = ctx["noun_part"] + new_clitic

        if error_word == position["original"]:
            return None

        return error_word
