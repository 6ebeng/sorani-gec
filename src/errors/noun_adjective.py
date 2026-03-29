"""
Noun-Adjective Agreement Error Generator

Injects noun-adjective agreement errors into Sorani Kurdish. Based on the
NP-internal agreement analysis in Slevanayi (2001) "Agreement in the Kurdish
Language" (ڕێککەوتن لە زمانی کوردیدا), pp. 37-48, which covers agreement
within the noun phrase (فریزی ناوی) including determiners, number,
definiteness, ezafe, and case.

Key findings from Slevanayi (2001) implemented here:

  1. ADJECTIVE INVARIANCE [F#79] (pp. 38-40): Adjectives in Kurdish are
     INVARIANT — they do NOT agree with the head noun in number or gender.
     This means a realistic error is to INCORRECTLY inflect an adjective
     for number (e.g., adding plural marker to an adjective), which a
     learner might do by analogy with other languages.

  2. DETERMINER-NOUN AGREEMENT [F#67, F#70] (pp. 41-44): Determiners
     (demonstratives, definite markers) agree with the head noun in number
     and, in oblique case, in gender. The key distinction is:
       - Nominative case: only NUMBER agreement on definite markers
       - Oblique case: NUMBER + GENDER agreement on determiners

  3. NOUN TYPE CONSTRAINTS [F#68, F#69, F#86] (pp. 45-48):
       - Proper nouns [F#86]: no singular/plural morphological marking
       - Mass nouns [F#68]: no plural form (always singular)
       - Collective nouns [F#69]: morphologically singular but semantically
         plural (trigger PLURAL agreement on the verb — Slevanayi p. 48)

  4. NO DEFINITENESS AGREEMENT [F#90] (pp. 47-48): There is NO agreement
     in definiteness/indefiniteness between head noun and modifiers. The
     definiteness marker ەکە applies to the head noun only. Demonstratives
     and quantifiers NEVER co-occur with indefinite marker ەک:
       *ئەوەک ✗, *هەندکەک ✗, *ئازادەک (proper noun) ✗
     Source: Slevanayi (2001), pp. 47-48.

  5. EZAFE SYSTEM [F#10, F#49, F#59, F#60]: The ezafe particle (ی) links
     nouns to their modifiers. Fatah & Qadir (2006) classify it as a
     single-morph single-morpheme unit. Errors include: deletion, doubling,
     or confusion with ی-ending nouns vs. ezafe.
     F#10 — Ezafe connects head noun + modifier in NP (Haji Marf 2014)
     F#49 — Ezafe and pronoun set interactions
     F#59 — Ezafe deletion (ی omission error)
     F#60 — Ezafe allomorphs: ی after consonant, بـ after ی-final nouns

  6. QUANTIFIER / NUMERAL NON-AGREEMENT [F#77] (Slevanayi pp. 87-88;
     Maaruf 2010, p. 139): Quantifiers and numerals force plural verb
     agreement while the noun remains morphologically singular. The noun
     in a quantified NP does NOT take the plural suffix, creating a surface
     mismatch that is a common source of confusion.

  7. CROSS-DIALECTAL EZAFE CONTRAST [F#2] (Maaruf 2010, pp. 109-117):
     In Sorani, the ezafe is gender-neutral, but in Kurmanji it is
     gender-sensitive (a/ê for masc/fem). This is relevant for error
     generation because Sorani writers influenced by Kurmanji may attempt
     gender inflection on the ezafe — a realistic error in bilectal
     communities.

  Additional findings referenced:
    F#115 — Definiteness marker migration (Farhadi 2013)
    F#117 — Pre-head determiners without ezafe
    F#131 — Derivational-before-grammatical affix ordering
    F#136 — Definiteness precedes all grammatical suffixes
    F#143 — Definiteness in coordinated vs. ezafe NPs
    F#165 — ی/یی double-ی scenarios
    F#182 — Attributive adjective cannot carry determiners
    F#186 — Determiner allomorphs (phonological conditioning)

Error strategies:
  A. Ezafe deletion: removing ezafe between noun and adjective
  B. Ezafe insertion: adding ezafe where it should not appear
  C. Definiteness mismatch: swapping sg↔pl definite/indefinite suffixes
  D. Adjective false inflection: ADDING plural marker to an adjective
     (exploiting the invariance rule — learners may over-generalise)
  E. Collective noun number flip: treating collective noun as plural
"""

import re
from typing import Optional

from .base import BaseErrorGenerator
from ..morphology.constants import (
    ADJECTIVE_DIMINUTIVE_SUFFIXES,
    CHAIN_ADJECTIVE_LAST_TAKES_MARKER,
    DEFINITE_ATTACHMENT_TYPES,
    DEFINITE_BLOCKS_SECONDARY_PLURAL,
    DEMONSTRATIVE_BLOCKS_PROPER_NOUN,
    IZAFE_E_BLOCKS_INDEFINITE_ADJECTIVE,
    PROPER_NOUN_BLOCKS_DEFINITE,
    SECONDARY_PLURAL_MARKERS,
)


# Definiteness suffixes in Sorani Kurdish
# Source: Slevanayi (2001), pp. 41-44 — NP-internal agreement
# DEFINITE_ATTACHMENT_TYPES (F#268, Haji Marf): 4 phonological types,
# DEFINITE_BLOCKS_SECONDARY_PLURAL (F#302, Haji Marf): definite suffix
# blocks secondary plural markers (-ات + -ان stacking).
DEFINITE_SUFFIXES = {
    "sg_def": "ەکە",        # the (singular): کتێبەکە (the book)
    "pl_def": "ەکان",       # the (plural): کتێبەکان (the books)
    "sg_indef": "ێک",       # a (singular): کتێبێک (a book)
    "pl_indef": "انێک",     # some (plural): کتێبانێک (some books)
}

# Common Sorani adjectives — INVARIANT forms that never agree
# Source: Slevanayi (2001), pp. 38-40
COMMON_ADJECTIVES = [
    "گەورە",     # big
    "بچووک",     # small
    "باش",       # good
    "خراپ",      # bad
    "نوێ",       # new
    "کۆن",       # old
    "جوان",      # beautiful
    "درێژ",      # long/tall
    "کورت",      # short
    "گرنگ",      # important
    "ئاسان",     # easy
    "سەخت",      # hard/difficult
    "تازە",      # fresh/new
    "زۆر",       # many/much
    "کەم",       # few/little
    "ڕەش",       # black
    "سپی",       # white
    "سوور",      # red
    "زەرد",      # yellow
    "شین",       # blue/green
    "خۆش",       # nice/pleasant
    "بەرز",      # high/tall
    "نزم",       # low
]

# Plural suffix for false-inflection errors on adjectives
PLURAL_SUFFIX = "ان"

# Collective nouns: morphologically singular, semantically plural
# Source: Slevanayi (2001), p. 48; Kurdish Academy grammar (2018)
COLLECTIVE_NOUNS = [
    "خەڵک",     # people
    "ئایل",      # tribe/family group
    "خێزان",     # family
    "لەشکر",    # army
    "کۆمەڵ",    # group/society
    "جەماوەر",  # crowd/public
    "هێڵ",       # flock
    "پۆلیس",    # police (collective)
    "سوپا",     # military
    "گەل",       # nation/people
    "دەستە",    # team/group
    "تاک",       # individuals (collective sense)
    "مەڕ",       # sheep (collective)
    "نەوتچی",  # oil workers (collective)
    "هاوڵاتی",  # citizenry (collective)
]

# Proper nouns — cannot take indefinite (ەک) or plural (ان/ین) markers
# Source: Slevanayi (2001), pp. 43-44
# PROPER_NOUN_BLOCKS_DEFINITE (F#265, Haji Marf): proper nouns also block
# definite ەکە attachment.
# DEMONSTRATIVE_BLOCKS_PROPER_NOUN (F#182): demonstratives cannot modify
# proper nouns directly.
# Proper nouns have unique reference: *هەولێرەک, *سلێمانییان are ungrammatical.
PROPER_NOUNS = [
    "سلێمانی", "هەولێر", "کوردستان", "عێراق", "دهۆک", "کەرکووک",
    "بەغدا", "سۆران", "بادینان", "ئەربیل", "موسڵ", "ئامەد",
    "مەهاباد", "سنە", "ڕانیە", "چوارتا", "حەڵەبجە",
]

# Suffixes that are illegal on proper nouns
# Source: Slevanayi (2001), pp. 43-44
PROPER_NOUN_ILLEGAL_SUFFIXES = ["ەکە", "ەکان", "ێک", "ان", "ین", "یەکە", "انێک"]

# Mass nouns — no plural form
# Source: Slevanayi (2001), p. 46; expanded from Kurdish Academy (2018)
# Mass nouns NEVER take plural suffix directly. They need a measure word
# (پێوەر) to be quantified, and the measure word controls verb agreement
# (Slevanayi 2001, pp. 46-47, 53, 57).
# Example: من دوو پەرداخیت شیری ڤەخوارن — verb agrees with "two glasses"
MASS_NOUNS = [
    "ئاو",      # water
    "شیر",       # milk
    "خۆڵ",       # dust/earth
    "خوێن",     # blood
    "هەوا",     # air/weather
    "نان",       # bread
    "دارو",     # medicine
    "هەناسە",   # breath
    "نەفت",     # oil
    "ئاسن",     # iron
    "ئالتوون",  # gold
    "قوماش",    # fabric
    "پارە",     # money (uncountable sense)
    "گەنم",     # wheat — Slevanayi (2001), p. 46
    "برنج",      # rice — Slevanayi (2001), p. 47
]

# Measure words (پێوەر) — quantify mass nouns; control verb agreement
# Source: Slevanayi (2001), pp. 47, 53, 57
# Structure: measure word = دیارخراو (head), mass noun = دیارخەر (dependent)
MEASURE_WORDS = [
    "پەرداخ",   # glass/cup
    "لبا",       # heap/pile — Slevanayi (2001), p. 47
    "گلاس",     # glass (loanword)
    "کیلۆ",     # kilogram
    "تۆن",       # ton
    "لیتر",      # litre
    "پارچە",    # piece
    "دانە",     # unit/piece
    "بۆتل",     # bottle
    "تەشت",     # basin
    "گۆنی",     # sack
    "قاپ",       # bowl/plate
]

# Ezafe marker
EZAFE = "ی"

# Demonstrative + definite marker co-occurrence restriction
# Source: Wrya Omar Amin (1986), Finding #10 — Rule R4
# ناوی ئیشارە (ئەم...ە, ئەو...ە) CANNOT co-occur with ەکە or ێک
DEMONSTRATIVE_PREFIXES = ["ئەم", "ئەو"]
DEMONSTRATIVE_SUFFIX = "ە"  # closing demonstrative suffix
INCOMPATIBLE_WITH_DEM = ["ەکە", "ێک", "یەکە"]  # cannot co-occur

# Vowel-final plural allomorphy — Mamajalakayi, Finding #59
# Stems ending in vowel take یان (not ان): قوتابی → قوتابییان
VOWEL_FINALS = ["ی", "ێ", "ۆ", "وو", "وو"]

# Definite marker allomorphy — Mamajalakayi, Finding #60
# Stems ending in vowel take یەکە (not ەکە): قوتابی → قوتابییەکە
# (implemented below in generate_error)


class NounAdjectiveErrorGenerator(BaseErrorGenerator):
    """Generate noun-adjective agreement errors.

    Targets ezafe constructions, definiteness agreement, adjective
    invariance violations, and collective/mass noun mismatches within
    noun phrases. Based on Slevanayi (2001), pp. 37-48.
    """

    @property
    def error_type(self) -> str:
        return "noun_adjective_agreement"

    def find_eligible_positions(self, sentence: str) -> list[dict]:
        """Find noun-adjective constructions where agreement can be broken.

        Detects four pattern types:
        1. Ezafe constructions: noun + ی + adjective
        2. Definite nouns: words ending in ەکە/ەکان
        3. Adjective + ezafe sequences: where false inflection can be injected
        4. Collective/mass nouns that may trigger number mismatch
        """
        positions = []

        # Pattern 1: Noun + ezafe (ی) + adjective
        # Source: Fatah & Qadir (2006) — ezafe as single-morph morpheme
        ezafe_pattern = re.compile(
            r'(?:^|(?<=\s))(\S+)(ی\s+)(\S+)(?=\s|$)',
        )

        for match in ezafe_pattern.finditer(sentence):
            noun = match.group(1)
            ezafe = match.group(2)
            adjective = match.group(3)

            # Only consider positions where the modifier is a known adjective
            is_adjective = adjective in COMMON_ADJECTIVES
            # Analyzer-enhanced adjective detection: when the hardcoded
            # list misses a word, fall back to the morphological analyzer.
            if not is_adjective and self.analyzer is not None:
                try:
                    adj_feats = self.analyzer.analyze_token(adjective)
                    is_adjective = adj_feats.pos == "ADJ"
                except Exception:
                    pass
            if not is_adjective and len(adjective) < 3:
                continue

            positions.append({
                "start": match.start(),
                "end": match.end(),
                "original": match.group(0),
                "context": {
                    "noun": noun,
                    "ezafe": ezafe,
                    "adjective": adjective,
                    "is_known_adjective": is_adjective,
                    "pattern_type": "ezafe",
                },
            })

        # Pattern 2: Definite noun (ەکە/ەکان) — can swap definiteness
        def_pattern = re.compile(
            r'(\S+?)(ەکە|ەکان)(?=\s|$)'
        )

        for match in def_pattern.finditer(sentence):
            stem = match.group(1)
            def_suffix = match.group(2)

            if len(stem) < 2:
                continue

            # Avoid overlap with ezafe matches
            overlap = any(
                not (match.end() <= p["start"] or match.start() >= p["end"])
                for p in positions
            )
            if overlap:
                continue

            # Check if stem is a collective or mass noun
            is_collective = stem in COLLECTIVE_NOUNS
            is_mass = stem in MASS_NOUNS

            # Track underlying number from the suffix for later validation
            underlying_number = "sg" if def_suffix == "ەکە" else "pl"

            positions.append({
                "start": match.start(),
                "end": match.end(),
                "original": match.group(0),
                "context": {
                    "stem": stem,
                    "suffix": def_suffix,
                    "is_collective": is_collective,
                    "is_mass": is_mass,
                    "underlying_number": underlying_number,
                    "pattern_type": "definiteness",
                },
            })

        # Pattern 3: Demonstrative + bare noun — correct form to corrupt (Rule R4)
        # Source: Wrya Omar Amin (1986) — ئەم/ئەو cannot co-occur with ەکە/ێک
        # We find the CORRECT form (ئەم/ئەو + noun + ە) and inject an
        # incompatible definite marker to corrupt it.
        dem_pattern = re.compile(
            r'(?:^|(?<=\s))(ئەم|ئەو)\s+(\S+?)(ە)(?=\s|$)'
        )
        for match in dem_pattern.finditer(sentence):
            stem = match.group(2)
            # Skip very short stems and avoid overlap
            if len(stem) < 2:
                continue
            overlap = any(
                not (match.end() <= p["start"] or match.start() >= p["end"])
                for p in positions
            )
            if overlap:
                continue
            positions.append({
                "start": match.start(),
                "end": match.end(),
                "original": match.group(0),
                "context": {
                    "demonstrative": match.group(1),
                    "stem": stem,
                    "dem_suffix": match.group(3),
                    "pattern_type": "det_cooccurrence",
                },
            })

        # Pattern 4: Plural allomorphy — vowel-final stems (Finding #59)
        # Correct: قوتابییان; Error: قوتابیان (missing ی)
        plural_pattern = re.compile(r'(\S+[یێۆ])(ان)(?=\s|$)')
        for match in plural_pattern.finditer(sentence):
            stem = match.group(1)
            if len(stem) < 2:
                continue
            overlap = any(
                not (match.end() <= p["start"] or match.start() >= p["end"])
                for p in positions
            )
            if overlap:
                continue
            positions.append({
                "start": match.start(),
                "end": match.end(),
                "original": match.group(0),
                "context": {
                    "stem": stem,
                    "suffix": "ان",
                    "pattern_type": "vowel_plural",
                },
            })

        # Pattern 5: Mass nouns — should NEVER take plural suffix directly.
        # Source: Slevanayi (2001), pp. 46-47, 53, 57
        # Find bare mass nouns in clean text and mark them for incorrect
        # pluralisation (error generation adds a plural suffix).
        for mass_noun in MASS_NOUNS:
            mass_bare_pattern = re.compile(
                rf'(?:^|(?<=\s))({re.escape(mass_noun)})(?=\s|$)'
            )
            for match in mass_bare_pattern.finditer(sentence):
                # Skip if a plural suffix is already attached (the regex
                # matched a substring of a longer word)
                end = match.end()
                remainder = sentence[end:end + 4]
                if remainder.startswith("ان") or remainder.startswith("ەکان"):
                    continue

                overlap = any(
                    match.start() >= p["start"] and match.start() < p["end"]
                    for p in positions
                )
                if overlap:
                    continue

                positions.append({
                    "start": match.start(),
                    "end": match.end(),
                    "original": match.group(0),
                    "context": {
                        "stem": match.group(1),
                        "pattern_type": "mass_noun_plural",
                    },
                })

        # Pattern 6: Proper nouns — should NEVER take indefinite/plural markers.
        # Source: Slevanayi (2001), pp. 43-44
        # Find bare proper nouns in clean text and mark them for incorrect
        # suffix attachment (error generation adds an illegal suffix).
        for proper in PROPER_NOUNS:
            pn_pattern = re.compile(
                rf'(?:^|(?<=\s))({re.escape(proper)})(?=\s|$)'
            )
            for match in pn_pattern.finditer(sentence):
                end = match.end()
                # Skip if already suffixed
                rest = sentence[end:end + 4]
                if any(rest.startswith(s) for s in PROPER_NOUN_ILLEGAL_SUFFIXES):
                    continue

                overlap = any(
                    match.start() >= p["start"] and match.start() < p["end"]
                    for p in positions
                )
                if overlap:
                    continue

                positions.append({
                    "start": match.start(),
                    "end": match.end(),
                    "original": match.group(0),
                    "context": {
                        "stem": match.group(1),
                        "pattern_type": "proper_noun_marker",
                    },
                })

        # Pattern 7: Chained adjective marker placement [F#318, Haji Marf].
        # CHAIN_ADJECTIVE_LAST_TAKES_MARKER: in a chain of adjectives
        # (noun-ی adj-ی adj), only the LAST adjective takes the
        # definiteness/number marker. Error: marker on a non-final adjective.
        # F#318: CHAIN_ADJECTIVE_FREE_ORDERING — adjective ordering in
        # chains is free (unlike English); no ordering error injected.
        # F#318: CHAIN_ADJECTIVE_WA_SUBSTITUTION — و can replace listing
        # in chains (ڕەش و سپی instead of ڕەشی سپی).
        if CHAIN_ADJECTIVE_LAST_TAKES_MARKER:
            chain_pat = re.compile(
                r'(?:^|(?<=\s))(\S+)ی\s+(\S+)ی\s+(\S+)(?=\s|$)'
            )
            for match in chain_pat.finditer(sentence):
                adj1 = match.group(2)
                adj2 = match.group(3)
                if adj1 in COMMON_ADJECTIVES and adj2 in COMMON_ADJECTIVES:
                    overlap = any(
                        not (match.end() <= p["start"] or match.start() >= p["end"])
                        for p in positions
                    )
                    if not overlap:
                        positions.append({
                            "start": match.start(),
                            "end": match.end(),
                            "original": match.group(0),
                            "context": {
                                "noun": match.group(1),
                                "adj1": adj1,
                                "adj2": adj2,
                                "pattern_type": "chained_adjective",
                            },
                        })

        return positions

    def generate_error(self, position: dict) -> Optional[str]:
        """Generate agreement error based on the pattern type.

        Five strategies, weighted by pattern type:
          A. Ezafe: drop or duplicate the ezafe marker
          B. Ezafe + known adjective: falsely inflect the adjective for
             number (adding ان), violating the invariance rule
             (Slevanayi 2001, pp. 38-40)
          C. Definiteness: swap singular↔plural
          D. Collective/mass noun: add incorrect plural marker
        """
        ctx = position["context"]
        pattern_type = ctx["pattern_type"]

        if pattern_type == "ezafe":
            # If modifier is a known invariant adjective, sometimes
            # generate a false-inflection error (Strategy D)
            if ctx.get("is_known_adjective") and self.rng.random() < 0.35:
                # Strategy D: Falsely inflect invariant adjective for plural
                # e.g., "کتێبی باش" → "کتێبی باشان" (incorrect plural on adj)
                # Source: Slevanayi (2001), pp. 38-40 — adjectives are invariant
                error = ctx["noun"] + ctx["ezafe"] + ctx["adjective"] + PLURAL_SUFFIX
            else:
                strategy = self.rng.random()
                if strategy < 0.4:
                    # Strategy A: Drop the ezafe (ungrammatical)
                    error = ctx["noun"] + " " + ctx["adjective"]
                elif strategy < 0.7:
                    # Strategy B: Replace ezafe with definite marker
                    error = ctx["noun"] + "ەکەی " + ctx["adjective"]
                else:
                    # Strategy C: Double the ezafe
                    error = ctx["noun"] + "یی " + ctx["adjective"]

        elif pattern_type == "definiteness":
            suffix = ctx["suffix"]
            stem = ctx["stem"]

            if ctx.get("is_mass"):
                # Mass nouns should not take plural — inject plural error
                # Source: Slevanayi (2001), p. 46
                if suffix == "ەکە":
                    error = stem + "ەکان"  # incorrect pluralisation
                else:
                    return None
            elif ctx.get("is_collective"):
                # Collective nouns: morph. sg but semantically pl
                # Error: treat as morphologically plural
                # Source: Slevanayi (2001), p. 48
                if suffix == "ەکە":
                    error = stem + "ەکان"
                elif suffix == "ەکان":
                    error = stem + "ەکە"
                else:
                    return None
            else:
                # Regular definiteness swap: singular↔plural
                if suffix == "ەکە":
                    error = stem + "ەکان"
                elif suffix == "ەکان":
                    error = stem + "ەکە"
                else:
                    return None
        elif pattern_type == "det_cooccurrence":
            # Strategy: inject incompatible definite/indefinite marker
            # into a correct demonstrative NP.
            # Source: Wrya Omar Amin (1986), Rule R4
            # Correct: ئەم کتێبە → Error: *ئەم کتێبەکە
            dem = ctx["demonstrative"]
            stem = ctx["stem"]
            bad_suffix = self.rng.choice(INCOMPATIBLE_WITH_DEM)
            error = dem + " " + stem + bad_suffix

        elif pattern_type == "vowel_plural":
            # Strategy: drop the linking ی in vowel-final plurals
            # Correct: قوتابییان → Error: قوتابیان
            # Source: Mamajalakayi, Finding #59
            stem = ctx["stem"]
            # Insert extra ی to make it look correct, then we generate
            # error by removing it (swap correct→wrong)
            # The matched form may already be wrong; generate the other form
            if stem.endswith("ی"):
                # Could be correct (vowel+یان) or wrong (missing ی)
                # Generate wrong form: remove one ی before ان
                error = stem[:-1] + "ان"
            else:
                return None

        elif pattern_type == "mass_noun_plural":
            # Strategy: Incorrectly attach plural suffix to mass noun.
            # Source: Slevanayi (2001), pp. 46-47, 53, 57
            # Mass nouns NEVER take plural suffix directly.
            # e.g., "ئاو" → "ئاوەکان" or "ئاوان"
            stem = ctx["stem"]
            suffix = self.rng.choice(["ان", "ەکان"])
            error = stem + suffix

        elif pattern_type == "proper_noun_marker":
            # Strategy: Incorrectly attach indefinite/plural marker to
            # proper noun. Source: Slevanayi (2001), pp. 43-44.
            # e.g., "هەولێر" → "هەولێرێک" or "هەولێران"
            stem = ctx["stem"]
            suffix = self.rng.choice(PROPER_NOUN_ILLEGAL_SUFFIXES)
            error = stem + suffix

        elif pattern_type == "chained_adjective":
            # F#318: Only the LAST adjective takes the marker.
            # Error: put the marker on the first adjective instead.
            noun = ctx["noun"]
            adj1 = ctx["adj1"]
            adj2 = ctx["adj2"]
            # Pick a definiteness marker to misplace on adj1
            bad_marker = self.rng.choice(["ەکە", "ان"])
            error = noun + "ی " + adj1 + bad_marker + "ی " + adj2
        else:
            return None

        if error == position["original"]:
            return None

        return error
