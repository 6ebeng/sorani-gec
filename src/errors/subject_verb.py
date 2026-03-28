"""
Subject-Verb Number Disagreement Error Generator

Injects subject-verb number agreement errors into Sorani Kurdish sentences.
Implements Slevanayi's (2001) two-law agreement system:

  Law 1 — Subject-verb [F#1, F#66, F#94]: The verb agrees with the subject
  in person and number. Applies to intransitive verbs in all tenses AND
  transitive verbs in present/future tenses (Slevanayi 2001, p. 89).

  Law 2 — Object-verb (ergative) [F#14, F#66, F#80]: The verb agrees with
  the OBJECT in person and number. Applies to past transitive verbs only,
  due to Sorani Kurdish's split-ergative alignment (Slevanayi 2001,
  pp. 60-61).

The verb morphology templates are based on the conjugation tables in
Amin (2016) "Verb Grammar of the Kurdish Language", pp. 17-18, 51-52
[F#95, F#98], and the formal verb morpheme ordering from Maaruf (2010)
"Phrase Structure in Kurdish", pp. 78-89 [F#110]:

  Morpheme ordering in Sorani verb forms (Maaruf 2010):
    Intransitive past:  Root + Tense + Person(Subject)
    Intransitive pres:  Tense + Root + Person(Subject)
    Transitive past:    Root + Tense + Person(Subject) + Person(Object)
    Transitive present: Tense + Person(Object) + Root + Person(Subject)

  This reversal of subject/object clitic ordering between past and present
  is the morphosyntactic basis of Sorani's split-ergative alignment [F#22].

Note on the 3SG ات allomorph [F#96]: Only 8 verbs use ات instead of ێت for
the 3rd-person singular present ending (Amin 2016, pp. 21-22).
The ات is actually epenthetic ت, not a true pronoun [F#48, F#167].

Background verb morphology / valency findings (documented):
  F#3   Case roles (Fillmore framework) inform argument structure
  F#4   Middle verbs / causative restriction
  F#7   Verb classification by case frame (1/2/3-place verbs)
  F#32  Verb argument structure types (trans/intrans/ditrans)
  F#54  VP structure: transitive vs intransitive formulas (Maaruf)
  F#58  Verb valency types (0-valency through 3-valency)

Key findings implemented:
  F#1   Subject-verb agreement as primary agreement domain
  F#9   Three clitic sets (Set 1, Set 2, Set 3/possessive)
  F#13  Agentive vs patientive intransitive split
  F#14  Past transitive double clitic structure
  F#22  Clitic role switching by tense
  F#35  Two subtypes of intransitive verbs (ڕوودان distinction)
  F#39  Past transitive clitic placement
  F#66  Two agreement laws (Slevanayi 2001)
  F#72  Existential هەبوون three-way distinction
  F#73  Interrogative pronouns fully invariant
  F#75  Compound subject person hierarchy (1st > 2nd > 3rd)
  F#76  Vocative-imperative number agreement
  F#80  Split ergativity
  F#82  Bare noun person-only agreement
  F#85  Negation conjunction first-conjunct agreement
  F#88  Compound noun subjects force plural
  F#94  Nominative case unifies both laws (Amin 2016)
  F#96  3SG ات allomorph for 8 verbs
  F#98  8-tense conjugation formulas

Examples of errors:
  [Law 1 — present, subject-verb]
  - Singular → plural: "من دەچم" → "من دەچین" (I go → We go)
  - Plural → singular: "ئەوان دەچن" → "ئەوان دەچێت" (They go → He goes)

  [Law 1 — imperative]
  - Number flip: "بنووسە" → "بنووسن" (write-SG! → write-PL!)

  [Law 2 — past transitive, object-verb]
  - Object suffix flip: "من نووسیم" → "من نووسیمان" (I wrote-it → I wrote-us)
    (Here م marks 1sg agent, but we flip the object-agreement suffix)
"""

import re
from typing import Optional

from .base import BaseErrorGenerator


from ..morphology.constants import (
    AT_ALLOMORPH_STEMS,
    PAST_ENDINGS as PAST_VERB_ENDINGS,
    PRESENT_VERB_ENDINGS,
)

# Imperative verb endings (mood prefix ب- or negative مە-)
# Source: Amin (2016), pp. 34-35
IMPERATIVE_ENDINGS = {
    "2sg": ["ە"],               # write! (sing.)
    "2pl": ["ن"],               # write! (pl.)
}

# PAST_VERB_ENDINGS imported from morphology.constants as PAST_ENDINGS

# Mapping for number flips (works for both present and past paradigms)
NUMBER_FLIP = {
    "1sg": "1pl",
    "1pl": "1sg",
    "2sg": "2pl",
    "2pl": "2sg",
    "3sg": "3pl",
    "3pl": "3sg",
}

# Subject pronouns and their person/number
SUBJECT_PRONOUNS = {
    "من": "1sg",      # I
    "تۆ": "2sg",      # You (sg)
    "ئەو": "3sg",     # He/She
    "ئێمە": "1pl",    # We
    "ئێوە": "2pl",    # You (pl)
    "ئەوان": "3pl",   # They
}

# Bare-noun subject agreement: when a common noun appears without any
# determiner or plural marker, it agrees with the verb in PERSON only
# (3rd person). Number is unmarked → verb defaults to 3sg.
# Source: Slevanayi (2001), pp. 55-56
# This constrains error generation: a bare noun + 3sg verb is CORRECT,
# so we should not flip a 3sg verb when the subject is a bare noun.
BARE_NOUN_DEFAULT_PN = "3sg"

# Negation conjunction first-conjunct agreement
# Source: Slevanayi (2001), p. 61
# With نە...نە or یا...یا, the verb agrees with the FIRST conjunct's
# person/number only, NOT the compound resolution.
NEGATION_CONJUNCTIONS = {"نە", "یا"}

# Coordination conjunction — for detecting compound noun subjects
# Source: Slevanayi (2001), p. 61 — compound noun subjects (X و Y)
# always force plural verb.
COORDINATION_CONJUNCTION = "و"

# Negation prefixes that can appear before or combined with verb prefixes.
# Source: Amin (2016), p. 51
# نە- (past negation), نا- (present/future negation), مە- (imperative negation)
NEGATION_PREFIXES = ["نە", "نا", "مە"]

# Vocative suffixes — imperative verb must agree in number with vocative NP
# Source: Slevanayi (2001), pp. 16, 72-73
# ۆ = masculine singular vocative, ێ = feminine singular vocative,
# ینۆ = plural vocative
# Examples:
#   کوڕۆ وەرە (boy.VOC.SG + come.IMP.SG) — singular match
#   کوڕینۆ وەرن (boys.VOC.PL + come.IMP.PL) — plural match
#   *کوڕۆ وەرن — vocative sg + imperative pl = agreement error
VOCATIVE_SINGULAR_SUFFIXES = ["ۆ", "ێ"]
VOCATIVE_PLURAL_SUFFIX = "ینۆ"

# Existential هەبوون — three distinct uses with different agreement
# Source: Slevanayi (2001), pp. 75-77
# 1. بوونی هاتنەئارایی (becoming) — regular intransitive Law 1
# 2. هەبوون (existence) — agreement set agrees in both tenses
# 3. هەبوون (possession) — possessive set NEVER agrees, verb stays 3sg
# Possession-use verbs should not be targeted for subject-verb error
# injection because they are inherently invariant (3sg).
EXISTENTIAL_POSSESSION_MARKERS = ["مم", "ت", "ی", "مان", "تان", "یان"]

# Compound verb prefixes that appear between negation/mood and the stem.
# Source: Amin (2016), p. 62
COMPOUND_VERB_PREFIXES = ["وەر", "هەڵ", "لێ", "تێ", "دەر", "پێ"]

# Past-tense verb stems (transitive — verb agrees with OBJECT per Law 2)
# Source: Slevanayi (2001), pp. 60-61; Amin (2016), pp. 15, 37, 51;
#         Kurdish Academy grammar (2018), pp. 106–107, 144–156;
#         Rasul (2005), pp. 13–14
TRANSITIVE_PAST_STEMS = [
    "کرد", "گوت", "دیت", "نووسی", "خوێند", "کڕی", "فرۆشت",
    "برد", "خوارد", "دا", "گرت", "ناسی", "ویست",
    # Additional from Amin (2016, pp. 15, 37) — irregular/suppletive
    "نارد", "شکاند", "ڕشت", "لایەند", "فێرکرد", "شوشت",
    "دۆزی", "هاوشت", "بەخشی", "کوشت", "سووتاند",
    "چنی", "کوتا", "پوختکرد", "زانی", "توانی", "بینی",
    # Kurdish Academy grammar + Rasul (2005) — expanded
    "هێنا", "تاشی", "پێچا", "پاراست", "خست",
    "بەست", "بیست", "کێڵا", "پێوا", "سپارد",
    "ژمارد", "بژارد", "شارد",
    # Causative transitives (past stems)
    "سووتاند", "وەستاند", "خزاند", "مراند", "تەقاند",
    "ڕژاند", "کەواند", "نواند", "فراند", "بەزاند",
]

# Past-tense verb stems (intransitive — verb agrees with SUBJECT per Law 1)
# Source: Slevanayi (2001), pp. 60-61; Amin (2016), p. 37;
#         Rasul (2005), pp. 4-5; Wrya Amin (1986), p. 25
#
# Subdivided into AGENTIVE (volitional subject) and PATIENTIVE (non-volitional)
# per Rasul (2005, p. 4) — this distinction matters for imperative formation
# (patientive intransitives cannot form imperatives).
INTRANSITIVE_PAST_STEMS_AGENTIVE = [
    "چوو", "هات", "نیشت", "گەیشت", "بوو",
    "ڕۆیشت", "مایەوە", "خەوت", "نووست",
    # Rasul (2005), pp. 4–5; Wrya Amin detailed linguistics
    "گەڕا", "فڕی", "سووڕا", "وەستا", "گریا", "دانیشت",
    "هەستا", "پەڕی", "چەقی", "هەڵسا",
    # Compound-intransitive agentive stems
    "جوڵا", "جەنگا", "تۆرا", "پارا", "سووڕا",
]

INTRANSITIVE_PAST_STEMS_PATIENTIVE = [
    # Patientive (ڕوودان/مطاوعە) — subject is patient, no imperative possible.
    # Source: Rasul (2005, p. 4); Kurdish Academy grammar (2018), pp. 80–106
    "مرد", "کەوت",
    "خلیسکا", "ئاوابوو", "ڕووخا", "شەقا",
    "سووتا", "پسا", "خنکا", "ڕژا", "شکا", "دڕا",
    "بڕا", "ترسا", "کرتا", "ڕما", "تکا", "تاسا",
    "بزڕکا", "ترشا", "قلیشا", "پچڕا", "دامرکا",
    "فەوتا", "پشکوتا", "خزی", "تەقی", "بزی",
    "کۆکی", "پژمی", "برژا", "ژیا",
]

# Combined for backward compatibility
INTRANSITIVE_PAST_STEMS = (
    INTRANSITIVE_PAST_STEMS_AGENTIVE + INTRANSITIVE_PAST_STEMS_PATIENTIVE
)

# Irregular present stems — verbs with suppletive present forms
# Source: Rasul (2005, p. 22); Kurdish Academy grammar, pp. 167–181
# Map: past_stem → present_stem (for stem-form error detection)
IRREGULAR_PRESENT_STEMS = {
    "ڕۆیشت": "ڕۆ",     # go
    "برد": "بە",        # carry
    "کرد": "کە",        # do
    "خوارد": "خۆ",      # eat
    "دا": "دە",         # give
    "شوشت": "شۆ",       # wash
    "هاوشت": "هاوێژ",   # throw
    "هێنا": "هێن",      # bring
    "کوشت": "کوژ",      # kill
    "گرت": "گر",        # take
    "گوت": "ڵێ",        # say
    "بیست": "بیس",      # hear
    "دیت": "بین",       # see
}

# ڕوودان (happening/non-volitional) verbs use ێ for ALL persons in present
# (not just 3sg). Source: Kurdish Academy grammar (2018), pp. 80–106, Finding #35
# Example: دەسووتێم, دەسووتێی, دەسووتێ (ێ in every person)
RUUDAN_PRESENT_STEMS = [
    "سووتێ", "پسێ", "خنکێ", "ڕژێ", "شکێ", "دڕێ",
    "بڕێ", "ترسێ", "کرتێ", "تکێ", "پژمێ", "خزێ",
    "تەقێ", "بزێ", "قلیشێ", "فەوتێ", "برژێ", "ژیێ",
]

# Causative verb stems (derived from intransitives via -(ا)ندن)
# Past tense uses ا vowel; present tense ا→ێ
# Source: Rasul (2005), pp. 23–24; Kurdish Academy grammar, pp. 96–106
CAUSATIVE_PAST_STEMS = [
    "سووتاند", "وەستاند", "خزاند", "مراند", "تەقاند",
    "ڕژاند", "شکاند", "بزاند", "کەواند", "نواند",
    "فراند", "بەزاند", "وەراند", "پساند",
]

# Suppletive causative pairs (no -اندن; lexically different stem)
# Source: Kurdish Academy grammar (2018), pp. 106–107
SUPPLETIVE_CAUSATIVE_PAIRS = {
    "کەوتن": "خستن",    # fall → drop
    "هاتن": "هێنان",    # come → bring
    "چوون": "بردن",     # go → take
    "ڕۆیشتن": "ناردن",  # leave → send
    "ڕژان": "ڕشتن",     # spill → pour
    "گۆڕان": "گۆڕین",   # change(intr) → change(tr)
    "مان": "هێشتن",     # stay → leave behind
}


class SubjectVerbErrorGenerator(BaseErrorGenerator):
    """Generate subject-verb number disagreement errors.

    Handles three verb categories per Slevanayi's (2001) two-law system:
      1. Present/future verbs (Law 1): flip the subject-agreement suffix
      2. Imperative verbs (Law 1): flip 2sg ↔ 2pl
      3. Past transitive verbs (Law 2): flip the object-agreement suffix
         (the ergative clitic), since the verb agrees with the object
    """

    @property
    def error_type(self) -> str:
        return "subject_verb_number"

    def find_eligible_positions(self, sentence: str) -> list[dict]:
        """Find verb positions where number can be flipped.

        Handles four verb patterns based on Amin (2016) and Slevanayi (2001):
        1. Present-tense: prefix (دە-/ئە-) + stem + person/number ending  [Law 1]
        2. Negated present: (نا-) + prefix (دە-) + stem + ending           [Law 1]
        3. Imperative: prefix (ب-/مە-) + stem + 2sg/2pl ending             [Law 1]
        4. Past-tense transitive: (neg?) stem + clitic ending               [Law 2]
        """
        positions = []

        # Build alternation for compound prefixes (وەر|هەڵ|لێ|تێ|دەر|پێ)
        compound_alt = "|".join(re.escape(p) for p in COMPOUND_VERB_PREFIXES)

        # Pattern 1: Present-tense verbs (with optional negation)
        # Matches: دەچم, نادەچم, ئەچم, نەدەچم, هەڵدەکەوم, etc.
        present_pattern = re.compile(
            rf'((?:نە|نا)?(?:{compound_alt})?(?:دە|ئە))(\w+?)(م|یت|ی|ێت|ێ|ات|ین|ن|ەم|ەن)(?=\s|$)'
        )

        for match in present_pattern.finditer(sentence):
            prefix = match.group(1)
            stem = match.group(2)
            ending = match.group(3)

            current_pn = self._identify_person_number(ending)
            if current_pn is None:
                continue

            positions.append({
                "start": match.start(),
                "end": match.end(),
                "original": match.group(0),
                "context": {
                    "prefix": prefix,
                    "stem": stem,
                    "ending": ending,
                    "person_number": current_pn,
                    "verb_type": "present",
                    "agreement_law": "law1",
                },
            })

        # Pattern 2: Imperative verbs (ب- or مە- prefix + stem + ە/ن)
        # Source: Amin (2016), pp. 34-35
        imperative_pattern = re.compile(
            rf'((?:{compound_alt})?(?:ب|مە))(\w+?)(ە|ن)(?=\s|$)'
        )

        for match in imperative_pattern.finditer(sentence):
            prefix = match.group(1)
            stem = match.group(2)
            ending = match.group(3)

            if ending == "ە":
                current_pn = "2sg"
            elif ending == "ن":
                current_pn = "2pl"
            else:
                continue

            # Avoid overlap with present-tense matches
            overlap = False
            for pos in positions:
                if (match.start() >= pos["start"] and match.start() < pos["end"]):
                    overlap = True
                    break
            if overlap:
                continue

            positions.append({
                "start": match.start(),
                "end": match.end(),
                "original": match.group(0),
                "context": {
                    "prefix": prefix,
                    "stem": stem,
                    "ending": ending,
                    "person_number": current_pn,
                    "verb_type": "imperative",
                    "agreement_law": "law1",
                },
            })

        # Pattern 3: Past-tense transitive verbs (Law 2 — ergative)
        # Source: Slevanayi (2001), pp. 60-61; Amin (2016), pp. 51-52
        # In past transitive, the verb agrees with the OBJECT. The clitic
        # suffixes (م/ت/∅/مان/تان/یان) on the verb mark the object.
        # Negation prefix نە- can appear before past stems.
        neg_alt = "|".join(re.escape(p) for p in ["نە"])
        past_endings_alt = "مان|تان|یان|م|ت"  # longest first; 3sg = zero

        for past_stem in TRANSITIVE_PAST_STEMS:
            # Match: (optional neg)(optional compound)(stem)(optional ending)
            pattern = re.compile(
                rf'(?:^|(?<=\s))((?:{neg_alt})?(?:{compound_alt})?)({re.escape(past_stem)})({past_endings_alt})?(?=\s|$)'
            )

            for match in pattern.finditer(sentence):
                prefix = match.group(1)
                stem = match.group(2)
                ending = match.group(3) or ""

                # Determine current person/number from ending
                current_pn = self._identify_past_person_number(ending)

                # Avoid overlap with existing positions
                overlap = False
                for pos in positions:
                    if (match.start() >= pos["start"] and match.start() < pos["end"]):
                        overlap = True
                        break
                    if (pos["start"] >= match.start() and pos["start"] < match.end()):
                        overlap = True
                        break
                if overlap:
                    continue

                positions.append({
                    "start": match.start(),
                    "end": match.end(),
                    "original": match.group(0),
                    "context": {
                        "prefix": prefix,
                        "stem": stem,
                        "ending": ending,
                        "person_number": current_pn,
                        "verb_type": "past_transitive",
                        "agreement_law": "law2",
                    },
                })

        return positions

    def generate_error(self, position: dict) -> Optional[str]:
        """Flip the verb's number (singular ↔ plural).

        For present and imperative verbs (Law 1): flips subject-agreement.
        For past transitive verbs (Law 2): flips object-agreement suffix
        (the ergative clitic), since the verb agrees with the object.
        """
        ctx = position["context"]
        current_pn = ctx["person_number"]
        verb_type = ctx.get("verb_type", "present")

        if current_pn not in NUMBER_FLIP:
            return None

        target_pn = NUMBER_FLIP[current_pn]

        if verb_type == "past_transitive":
            # Law 2: flip the object-agreement clitic (past paradigm)
            target_ending = PAST_VERB_ENDINGS.get(target_pn, "")
            error_verb = ctx["prefix"] + ctx["stem"] + target_ending
        elif verb_type == "imperative":
            target_endings = IMPERATIVE_ENDINGS.get(target_pn, [])
            if not target_endings:
                return None
            new_ending = self.rng.choice(target_endings)
            error_verb = ctx["prefix"] + ctx["stem"] + new_ending
        else:
            # Law 1: present/future — flip subject-agreement suffix
            target_endings = list(PRESENT_VERB_ENDINGS.get(target_pn, []))
            if not target_endings:
                return None
            # Filter ات vs ێت based on stem: only AT-allomorph verbs
            # (F#96) use ات for 3sg; all others use ێت/ێ.
            stem = ctx.get("stem", "")
            is_at_verb = any(stem.endswith(s) for s in AT_ALLOMORPH_STEMS)
            if target_pn == "3sg":
                if is_at_verb:
                    target_endings = [e for e in target_endings if e != "ێت"]
                else:
                    target_endings = [e for e in target_endings if e != "ات"]
                if not target_endings:
                    return None
            new_ending = self.rng.choice(target_endings)
            error_verb = ctx["prefix"] + ctx["stem"] + new_ending

        # Don't return if identical to original
        if error_verb == position["original"]:
            return None

        return error_verb

    def _identify_person_number(self, ending: str) -> Optional[str]:
        """Identify person/number from present-tense verb ending."""
        for pn, endings in PRESENT_VERB_ENDINGS.items():
            if ending in endings:
                return pn
        return None

    def _identify_past_person_number(self, ending: str) -> str:
        """Identify person/number from past-tense clitic ending."""
        if not ending:
            return "3sg"  # zero morpheme = 3rd person singular
        for pn, suffix in PAST_VERB_ENDINGS.items():
            if ending == suffix:
                return pn
        return "3sg"
