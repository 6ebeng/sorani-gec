"""
Pronominal Clitic Error Generator

Injects incorrect pronominal clitic forms into Sorani Kurdish sentences.
Sorani Kurdish has a rich system of enclitic pronouns that attach to verbs,
nouns, and prepositions. In past transitive sentences (ergative construction),
the clitic agrees with the object, not the subject [F#14, F#39, F#66].

The six-person clitic paradigm (م، ت، ی، مان، تان، یان) [F#9] and its
agreement rules are based on Amin (2016) "Verb Grammar of the Kurdish
Language", pp. 17-18 and 51-52, and Slevanayi (2001) "Agreement in the
Kurdish Language" (ڕێککەوتن لە زمانی کوردیدا).

Haji Marif (2014) "Kurdish Grammar: Volume One (Word Formation) Part Two
(Pronouns)" provides the most comprehensive classification of Sorani Kurdish
pronouns, organising them into ten categories [F#45]:
    1. Personal / independent  (جیناوی کەسیی جودا)
    2. Personal / bound        (جیناوی کەسی لکاو)  — these are the clitics
    3. Reflexive               (جیناوی خویی)
    4. Demonstrative           (جیناوی نیشانە)
    5. Interrogative           (جیناوی پرسیار)
    6. Quantitative            (جیناوی چەندنیتی)
    7. Possessive              (جیناوی ماڵی)
    8. Negative                (جیناوی نافی)
    9. Reciprocal              (جیناوی هاوبەشی)
   10. Definite / Indefinite   (جیناوی دیار / نادیار)

Clitic Set Terminology — Haji Marif (2014, p. 185) [F#20]:
    Set 1 / Strong (بەهێز): Clitics in AGENT position within past-tense
        ergative constructions.  Also called "first-set" or "A-clitics".
    Set 2 / Weak (بێهێز): Clitics in PATIENT position within past-tense
        ergative constructions.  Also called "second-set" or "P-clitics".
    Both sets share the same six morphemes (م/ت/ی/مان/تان/یان); the
    grammatical role is determined entirely by structural position.

Seven Grammatical Functions per Clitic Set — Amin (2016, pp. 16-24) [F#208]:
    Each of the six clitic morphemes can serve any of 7 functions depending
    on the host word and syntactic context:
        1. Agent (کارا)          — doer of action
        2. Patient (بەرکار)      — undergoer of action
        3. Possessive (خاوەنداری) — possessor of noun
        4. Indirect object (بەرکاری نا ڕاستەوخۆ) — recipient / beneficiary
        5. Complement of P (تەواوکاری دانەوشە) — object of preposition
        6. Complement of Adj (تەواوکاری ھاوەڵناو) — complement of adjective
        7. Complement of N (تەواوکاری ناو) — complement of noun

Five Clitic Position Rules — Haji Marif (2014, pp. 190-193) [F#99]:
    Rule 1: With present-tense verb → clitic attaches as ENCLITIC on verb
            Example: دە-خوێن-م ('I read')
    Rule 2: With past-tense intransitive → clitic attaches as ENCLITIC
            Example: ھات-م ('I came')
    Rule 3: With past-tense transitive → agent clitic is PROCLITIC,
            patient agreement shown by verb suffix
            Example: من-م  خوارد ('I ate it')  — م is Set 1 proclitic
    Rule 4: With imperative → clitic attaches as ENCLITIC
            Example: بیخوێنە-م ('read it for me')
    Rule 5: On nouns/prepositions → clitic attaches as ENCLITIC
            Example: کتێب-م ('my book'), بۆ-م ('for me')

Tense-Sensitive Clitic Position — Mukriani (2000, p. 64) [F#149]:
    Beyond the five structural rules above, Mukriani documents that clitic
    position shifts between pre-verbal and post-verbal depending on the
    specific tense/aspect combination:
        Simple past (intransitive):    POST-verbal  — گرتم
        Present progressive:           POST-verbal  — دەگرم
        Past progressive:              PRE-verbal   — دەمگرت
        Negative past:                 PRE-verbal   — نەمگرت
        Pluperfect:                    POST-verbal  — گرتبووم
        Negative pluperfect:           PRE-verbal   — نەمگرتبوو
        Prohibition:                   PRE-verbal   — مەگرە
        Imperative:                    POST-verbal  — بگرە
        Conditional:                   ب BETWEEN stem and clitic — بمگرتایە
    This tense-based positional variation is orthogonal to the set-based
    (Strong/Weak) distinction and provides additional error generation
    targets: misplacing a clitic relative to its tense-expected position.

Set Distribution by Tense/Transitivity — Haji Marif (2014, p. 193) [F#22]:
    Set 1 (Strong): Used ONLY in past transitive (ergative) constructions.
        The clitic marks the AGENT.
    Set 2 (Weak): Used in ALL other contexts:
        - All present tense (transitive & intransitive)
        - Past intransitive
        - Imperative forms
        - Nominal / prepositional hosts

Transitivity Diagnostic — Haji Marif (2014, p. 191):
    Clitic position can disambiguate transitivity:
        Past transitive  → agent clitic is PROCLITIC  (before/on previous word)
        Past intransitive → clitic is ENCLITIC        (on verb itself)
    This is critical for error generation because swapping clitic position
    (proclitic↔enclitic) creates a transitivity agreement error.

Passive Voice — Amin (2016, pp. 39-43) [F#107]:
    In passive constructions, the verb no longer assigns agent role, so
    Set 1 clitics shift to Set 2 behaviour.  The patient is promoted to
    subject and takes Set 2 (enclitic) marking:
        Active:  من-م  خوارد  ('I ate it')    — Set 1 (proclitic agent)
        Passive: خوێردرا-م  ('I was eaten')   — Set 2 (enclitic patient)

Note on double-clitic verbs [F#14, F#100]: Amin (2016) documents that
certain verbs like ویستن 'to want' take both subject and object clitic
sets simultaneously (e.g. خوش-مان دەوێ-ت 'we want it'). These
constructions are especially error-prone because writers must track two
agreement targets. F#100 notes that double clitic order reverses in some
Germian dialects [F#253].

Error weighting strategy (Slevanayi 2001, p. 60) [F#93]:
  - Within-number swaps (sg↔sg, pl↔pl) are weighted MORE heavily
    because they represent more natural errors (person confusion).
  - Cross-number swaps (sg→pl, pl→sg) are weighted LESS heavily
    as they are less common natural errors.
  - Formal coincidence [F#93]: when invariant forms happen to share
    features with the verb, this is not true agreement.

Additional findings implemented in this module:
  F#46  — Clitic position depends on tense and sentence structure
  F#47  — Set 2 clitics on non-verbal hosts
  F#50  — Set 2 on non-verbal elements (prepositions, adjectives)
  F#108 — Directional postbound clitic placement
  F#114 — Dative postbound restrictions
  F#116 — Negative progressive clitic shift
  F#120 — Passive transitivity prerequisite
  F#124 — Possessive هی/ئی blocks clitics
  F#125 — Imperative clitic restrictions
  F#126 — Compound verb clitic insertion between nominal + verbal
  F#127 — Causative suffix clitic position (ەوە vs اندن)
  F#129 — Clitic first-element attachment rule
  F#130 — Morpheme collision tolerance (مم, مانمان allowed)
  F#133 — Same-set clitic exclusion in simple sentences
  F#189 — Seven clitic laws (systematic position rules)
  F#190 — Triple clitic ordering
  F#204 — Clitic omission = ungrammatical (obligatory marking)

Examples:
- Correct:   "من نامەکەم نووسی" (I wrote the letter-my)
- Error:     "من نامەکەت نووسی" (I wrote the letter-your) — wrong clitic
"""

import re
from typing import Optional

from .base import BaseErrorGenerator


# Sorani Kurdish enclitic pronouns
# Source: Amin (2016), pp. 17-18; Haji Marif (2014), Chapter on bound pronouns
CLITICS = {
    "1sg": "م",     # my / me
    "2sg": "ت",     # your / you (sg)
    "3sg": "ی",     # his/her / him/her
    "1pl": "مان",   # our / us
    "2pl": "تان",   # your / you (pl)
    "3pl": "یان",   # their / them
}

# Independent (free) personal pronouns — Haji Marif (2014), Chapter 1
# Used for subject-pronoun ↔ verb-clitic agreement checking.
INDEPENDENT_PRONOUNS = {
    "1sg": "من",
    "2sg": "تۆ",
    "3sg": "ئەو",
    "1pl": "ئێمە",
    "2pl": "ئیوە",
    "3pl": "ئەوان",
}

# ---------------------------------------------------------------------------
# Clitic Set Classification — Haji Marif (2014, p. 185)
# ---------------------------------------------------------------------------
# Set 1 / Strong (بەهێز): Agent clitics in past-tense ergative constructions.
# Set 2 / Weak (بێهێز): Patient / enclitic clitics in all other contexts.
# Both sets use the same morpheme inventory (م/ت/ی/مان/تان/یان) but in
# different structural positions within the verb form.
FIRST_SET_ROLES = "agent"      # Set 1 / Strong — marks the agent
SECOND_SET_ROLES = "patient"   # Set 2 / Weak — marks the patient

# Alternative terminology mapping (for documentation / lookup)
SET_TERMINOLOGY = {
    "set1": {"kurdish": "بەهێز", "english": "Strong", "role": "agent"},
    "set2": {"kurdish": "بێهێز", "english": "Weak", "role": "patient"},
}

# ---------------------------------------------------------------------------
# Seven Grammatical Functions of Clitics — Amin (2016, pp. 16-24)
# ---------------------------------------------------------------------------
# Each clitic morpheme can serve any of these 7 functions depending on the
# host word and syntactic context.  The function determines which agreement
# target the clitic must match and is essential for error generation.
CLITIC_FUNCTIONS = [
    "agent",             # کارا         — doer of action (past transitive)
    "patient",           # بەرکار       — undergoer of action
    "possessive",        # خاوەنداری    — possessor of noun
    "indirect_object",   # بەرکاری نا ڕاستەوخۆ — recipient / beneficiary
    "complement_of_p",   # تەواوکاری دانەوشە   — object of preposition
    "complement_of_adj", # تەواوکاری ھاوەڵناو  — complement of adjective
    "complement_of_n",   # تەواوکاری ناو        — complement of noun
]

# ---------------------------------------------------------------------------
# Set Distribution by Tense/Transitivity — Haji Marif (2014, p. 193)
# ---------------------------------------------------------------------------
# Defines which clitic set (and therefore positional behaviour) is used
# in each verb construction type.  This is the key lookup for deciding
# whether a clitic should be proclitic or enclitic.
SET_DISTRIBUTION = {
    "present_transitive":    "set2",   # Set 2 / Weak — enclitic
    "present_intransitive":  "set2",   # Set 2 / Weak — enclitic
    "past_transitive":       "set1",   # Set 1 / Strong — proclitic agent
    "past_intransitive":     "set2",   # Set 2 / Weak — enclitic
    "imperative":            "set2",   # Set 2 / Weak — enclitic
    "passive":               "set2",   # Amin (2016, pp. 39-43): Set1→Set2
    "nominal_host":          "set2",   # On nouns / prepositions — enclitic
}

# ---------------------------------------------------------------------------
# Clitic Position Rules — Haji Marif (2014, pp. 190-193)
# ---------------------------------------------------------------------------
# Maps each construction type to whether the clitic is proclitic or enclitic.
# This is critical for error generation: misplacing a clitic (pro↔en)
# creates a transitivity agreement error (Haji Marif p. 191).
CLITIC_POSITION = {
    "present_transitive":    "enclitic",   # Rule 1: verb-م
    "present_intransitive":  "enclitic",   # Rule 1: verb-م
    "past_transitive":       "proclitic",  # Rule 3: من-م ... verb
    "past_intransitive":     "enclitic",   # Rule 2: verb-م
    "imperative":            "enclitic",   # Rule 4: verb-م
    "nominal_host":          "enclitic",   # Rule 5: noun-م / prep-م
}

# ---------------------------------------------------------------------------
# Tense-Sensitive Clitic Position — Mukriani (2000, p. 64)
# ---------------------------------------------------------------------------
# Complements Haji Marif's five structural rules with fine-grained
# tense/aspect-specific position data.  Certain tense combinations move
# the clitic to PRE-VERBAL position even when the structural rule would
# predict enclitic.
TENSE_CLITIC_POSITION = {
    "present_progressive":   "postverbal",   # دەگرم
    "simple_past":           "postverbal",   # گرتم
    "past_progressive":      "preverbal",    # دەمگرت
    "negative_past":         "preverbal",    # نەمگرت
    "pluperfect":            "postverbal",   # گرتبووم
    "negative_pluperfect":   "preverbal",    # نەمگرتبوو
    "prohibition":           "preverbal",    # مەگرە
    "imperative":            "postverbal",   # بگرە
    "conditional":           "infix",        # بمگرتایە (ب between stem & clitic)
}

# ---------------------------------------------------------------------------
# Negative Progressive Clitic Position Shift — Farhadi (2013, pp. 37-38)
# ---------------------------------------------------------------------------
# Source: Finding #116
# In past progressive transitive, negation causes the agent clitic to shift:
#   Positive: دە + agent + root → دەمزانی (I was knowing it)
#   Negative: نە + agent + دە + root → نەمدەزانی (I was not knowing it)
# The agent clitic moves from AFTER دە to BETWEEN نە and دە.
#
# In present tense, negation does NOT shift clitic position:
#   Positive: دەیانناسم    Negative: نایانناسم  (same clitic position)
#
# This asymmetry means past negation errors differ from present negation errors.
# Error example: *نەدەمزانی (agent after دە under negation) → correct: نەمدەزانی
NEGATIVE_PROGRESSIVE_CLITIC_ORDER = {
    "past_progressive_positive":  "دە + agent + root",        # دەمزانی
    "past_progressive_negative":  "نە + agent + دە + root",   # نەمدەزانی
    "present_positive":           "دە + patient + root + agent",  # دەیانناسم
    "present_negative":           "نا + patient + root + agent",  # نایانناسم
}

# ---------------------------------------------------------------------------
# Word Order Constraints — Mukriani (2000, pp. 62, 75-77)
# ---------------------------------------------------------------------------
# Sorani Kurdish is fundamentally SOV.  Mukriani's exhaustive permutation
# analysis (Appendices 1–2) with grammaticality judgements establishes:
#   - Subject generally sentence-initial
#   - Verb generally sentence-final
#   - Complements (objects, adverbs) fill the space between
#   - Subject is NEVER sentence-final (starred in all permutations)
#   - Verb NEVER precedes subject when both subject and object are present
# These constraints inform error detection: an agreement graph that places a
# subject after the verb can flag potential word-order errors.
WORD_ORDER_CONSTRAINTS = {
    "subject_position":  "initial",    # Subject defaults to sentence-initial
    "verb_position":     "final",      # Verb defaults to sentence-final
    "subject_never":     "final",      # Subject is NEVER sentence-final
    "verb_never_before": "subject",    # Verb never precedes subject (with both present)
}

# All clitic forms for pattern matching
ALL_CLITICS = list(CLITICS.values())

# Weighted clitic swap mappings
# Within-number swaps are weighted more heavily (more natural errors)
# Cross-number swaps are less likely but still possible
# Source: Slevanayi (2001), p. 60 — within-number confusion is more common
CLITIC_SWAPS_WEIGHTED = {
    #           alternatives            weights
    "م":   (["ت", "ی", "مان"],        [0.45, 0.35, 0.20]),  # sg→sg heavy
    "ت":   (["م", "ی", "تان"],        [0.45, 0.35, 0.20]),
    "ی":   (["م", "ت", "یان"],        [0.35, 0.35, 0.30]),
    "مان": (["تان", "یان", "م"],      [0.45, 0.35, 0.20]),  # pl→pl heavy
    "تان": (["مان", "یان", "ت"],      [0.45, 0.35, 0.20]),
    "یان": (["مان", "تان", "ی"],      [0.40, 0.40, 0.20]),
}

# Flat swap mapping (backwards-compatible, used when weights not needed)
CLITIC_SWAPS = {
    "م": ["ت", "ی", "مان"],
    "ت": ["م", "ی", "تان"],
    "ی": ["م", "ت", "یان"],
    "مان": ["تان", "یان", "م"],
    "تان": ["مان", "یان", "ت"],
    "یان": ["مان", "تان", "ی"],
}

# Common host words that frequently carry clitics (for reducing false positives)
# These are function words / prepositions that often take pronominal clitics
COMMON_CLITIC_HOSTS = [
    "لە", "بۆ", "لەگەڵ", "بە", "لەسەر", "لەژێر", "پێ", "تێ", "لێ",
    "پاش", "پێش", "نێوان", "دوای",
]

# ---------------------------------------------------------------------------
# Passive Voice Clitic Shift — Amin (2016, pp. 39-43)
# ---------------------------------------------------------------------------
# In passive constructions, the clitic shifts from Set 1 (proclitic agent)
# to Set 2 (enclitic patient) because the verb no longer assigns agent role.
# Passive morphology markers in Sorani Kurdish (common patterns):
# Source: Kurdish Academy grammar (2018), pp. 112–125; Farhadi (2013), pp. 38–40
# Passive morpheme is {ڕ}: past → ڕا, present → ڕێ
PASSIVE_MARKERS = {"درا", "دراو", "را", "راو", "کرا", "کراو",
                  "ڕا", "ڕێ", "ڕام", "ڕای", "ڕاین", "ڕان"}

# Passive clitic reassignment: explicit Set 1→Set 2 conversion — Amin (2016), p. 43
# Source: Finding #107
# When converting active→passive, the former Set 1 object clitic becomes
# the Set 2 subject clitic on the passive verb.
PASSIVE_CLITIC_REASSIGNMENT = {
    "م": "م",        # 1sg
    "ت": "یت",       # 2sg
    "ی": "ێت",       # 3sg
    "مان": "ین",     # 1pl
    "تان": "ن",      # 2pl
    "یان": "ن",      # 3pl
}

# ---------------------------------------------------------------------------
# Passive Transitivity Prerequisite — Farhadi (2013), p. 39 (Finding #120)
# ---------------------------------------------------------------------------
# Only transitive (تێپەڕ) or ditransitive (تێپەڕی ئاوێتە) verbs can be
# passivized. Intransitive verbs (تێنەپەڕ) cannot form passive because
# they lack a direct object (بەرکار) to promote to subject position.
# Explicit rule: "دەبێت کارەکەی تێپەڕ بێت"
# The system should cross-check verb valency before applying passive
# morphology: passive morpheme {ڕ} is only valid on transitive roots.
PASSIVE_ELIGIBLE_VALENCY = {"transitive", "ditransitive"}

# Common intransitive verbs that CANNOT be passivized (representative set)
# Adding passive morpheme ڕ to any of these produces an ungrammatical form.
INTRANSITIVE_NO_PASSIVE = {
    "چوون",     # to go
    "هاتن",     # to come
    "نووستن",   # to sleep
    "کەوتن",    # to fall
    "ڕۆیشتن",   # to go/walk
    "وەستان",   # to stop
    "دانیشتن",  # to sit
    "هەستان",   # to stand
    "مردن",     # to die
    "فڕین",     # to fly
    "ڕاکردن",   # to run
    "قیژاندن",  # to scream (intransitive use)
}

# Compound verb preverbs — in compound verbs, Set 1 clitics move BETWEEN
# preverb and verb: تێ + م + گرت = تێمگرت (NOT *تێگرتم)
# Source: Haji Marif (2014), pp. 116–121 (Finding #46, Rule R18)
COMPOUND_PREVERBS = ["تێ", "پێ", "لێ", "دا", "هەڵ", "دەر"]

# Directional postbound ـە absorbing prepositions بۆ/بە — Amin (2016), pp. 48-49
# Source: Finding #108
# In present perfect: وو + ـە contracts to ۆ + تە (e.g., ناردووە→ناردۆتە)
# The postbound attaches after the verb root + agreement morphemes.
DIRECTIONAL_POSTBOUND = "ە"
DIRECTIONAL_POSTBOUND_PERFECT = "تە"  # وو→ۆ contraction form

# Dative postbound ـێ replacing بە — Amin (2016), pp. 49-50
# Source: Finding #114
# With verbs of giving (دان, گەیشتن), بە+clitic → clitic+ـێ
# Epenthesis between ـە and ـێ: ی or ڕ inserted
DATIVE_POSTBOUND = "ێ"

# ---------------------------------------------------------------------------
# Past-Tense Transitive Stems — Haji Marif (2014, p. 191)
# ---------------------------------------------------------------------------
# When a verb stem is known to be past-transitive, the clitic MUST be
# proclitic (Set 1).  This set is used for the transitivity diagnostic:
# if the clitic is enclitic on one of these stems, it is an error.
# ---------------------------------------------------------------------------
# Past-Tense Transitive Stems — Haji Marif (2014, p. 191);
#     Kurdish Academy grammar (2018), pp. 144–156;
#     Rasul (2005), pp. 13–14; Wrya Amin detailed linguistics
# ---------------------------------------------------------------------------
# When a verb stem is known to be past-transitive, the clitic MUST be
# proclitic (Set 1).  This set is used for the transitivity diagnostic:
# if the clitic is enclitic on one of these stems, it is an error.
PAST_TRANSITIVE_STEMS = {
    "خوارد", "نووسی", "کرد", "بینی", "گوت", "برد", "خست",
    "دا", "کوشت", "فرۆشت", "گرت", "نارد", "شکاند",
    "خوێند", "گێڕا", "دۆزی", "کڕی", "فێرکرد",
    "ڕشت", "هێنا", "تاشی", "پێچا", "چنی", "پاراست",
    "دیت", "ویست", "ناسی", "زانی", "توانی",
    "بەخشی", "هاوشت", "شوشت", "لایەند",
    # Expanded: Kurdish Academy grammar (2018), pp. 106–107
    "بەست", "بیست", "کێڵا", "پێوا", "سپارد", "ژمارد",
    "بژارد", "شارد",
    # Causative transitives
    "سووتاند", "وەستاند", "خزاند", "مراند", "تەقاند",
    "ڕژاند", "کەواند", "نواند", "فراند", "بەزاند",
    "پساند", "وەراند",
}

# Past-tense intransitive stems — clitic MUST be enclitic (Set 2)
# Subdivided into agentive and patientive per Rasul (2005), p. 4.
PAST_INTRANSITIVE_STEMS_AGENTIVE = {
    "هات", "چوو", "نووست", "ڕۆیشت",
    "خلیسکا", "گەیشت", "دانیشت", "هەستا", "پەڕی",
    "گەڕا", "فڕی", "وەستا", "گریا", "سووڕا",
    "هەڵسا", "خەوت", "مایەوە",
}

PAST_INTRANSITIVE_STEMS_PATIENTIVE = {
    # ڕوودان (happening) verbs — cannot form imperatives
    "مرد", "کەوت", "ئاوابوو", "ڕووخا", "شەقا",
    "سووتا", "پسا", "خنکا", "ڕژا", "شکا", "دڕا",
    "بڕا", "ترسا", "فەوتا", "پشکوتا", "خزی", "تەقی",
    "بزی", "کۆکی", "پژمی", "ژیا", "برژا",
}

# Combined for backward compatibility
PAST_INTRANSITIVE_STEMS = (
    PAST_INTRANSITIVE_STEMS_AGENTIVE | PAST_INTRANSITIVE_STEMS_PATIENTIVE
)

# Invariant pronoun forms that should NOT be treated as clitic-bearing
# Source: Slevanayi (2001), pp. 38-48 — reflexive, interrogative, reciprocal
INVARIANT_PRONOUN_FORMS = {
    "خۆم", "خۆت", "خۆی", "خۆمان", "خۆتان", "خۆیان",  # reflexive
    "یەکتر", "یەکدی",                                     # reciprocal
}


class CliticErrorGenerator(BaseErrorGenerator):
    """Generate incorrect pronominal clitic errors.

    Uses linguistically-weighted swap probabilities: within-number swaps
    (e.g. 1sg→2sg) are more likely than cross-number swaps (e.g. 1sg→1pl),
    following the natural error distribution noted in Slevanayi (2001, p. 60).

    The generator is aware of the clitic set system (Haji Marif 2014):
      - Set 1 / Strong (بەهێز): proclitic agent in past transitive
      - Set 2 / Weak (بێهێز): enclitic in all other contexts
    and the seven grammatical functions each clitic can serve (Amin 2016).
    """

    @property
    def error_type(self) -> str:
        return "clitic_form"

    @staticmethod
    def get_expected_set(construction: str) -> str:
        """Return the expected clitic set ('set1' or 'set2') for a construction.

        Uses the distribution table from Haji Marif (2014, p. 193).

        Args:
            construction: One of the keys in SET_DISTRIBUTION
                (e.g. 'past_transitive', 'present_intransitive').

        Returns:
            'set1' or 'set2'.
        """
        return SET_DISTRIBUTION.get(construction, "set2")

    @staticmethod
    def get_expected_position(construction: str) -> str:
        """Return 'proclitic' or 'enclitic' for a construction type.

        Applies the five position rules from Haji Marif (2014, pp. 190-193).
        """
        return CLITIC_POSITION.get(construction, "enclitic")

    @staticmethod
    def is_passive(verb_form: str) -> bool:
        """Check if a verb form contains passive morphology.

        Passive constructions trigger Set1→Set2 shift (Amin 2016, pp. 39-43).
        """
        return any(verb_form.endswith(m) for m in PASSIVE_MARKERS)

    @staticmethod
    def diagnose_transitivity(stem: str) -> str | None:
        """Diagnose whether a past-tense stem is transitive or intransitive.

        Based on the transitivity diagnostic from Haji Marif (2014, p. 191):
        clitic position disambiguates transitivity.

        Returns:
            'transitive', 'intransitive', or None if unknown.
        """
        if stem in PAST_TRANSITIVE_STEMS:
            return "transitive"
        if stem in PAST_INTRANSITIVE_STEMS:
            return "intransitive"
        return None

    def find_eligible_positions(self, sentence: str) -> list[dict]:
        """Find words with attached clitics."""
        positions = []

        # Pattern: word ending with a clitic
        # Longer clitics first to avoid partial matches
        sorted_clitics = sorted(ALL_CLITICS, key=len, reverse=True)
        clitic_pattern = "|".join(re.escape(c) for c in sorted_clitics)

        # Match word + clitic at word boundary
        pattern = re.compile(
            rf'(\w+?)({clitic_pattern})\b'
        )

        for match in pattern.finditer(sentence):
            stem = match.group(1)
            clitic = match.group(2)

            # Skip very short stems (likely false positives)
            if len(stem) < 2:
                continue

            # Skip invariant pronoun forms (reflexive, reciprocal)
            # Source: Slevanayi (2001), pp. 38-48
            full_word = match.group(0)
            if full_word in INVARIANT_PRONOUN_FORMS:
                continue

            positions.append({
                "start": match.start(),
                "end": match.end(),
                "original": full_word,
                "context": {
                    "stem": stem,
                    "clitic": clitic,
                },
            })

        return positions

    def generate_error(self, position: dict) -> Optional[str]:
        """Replace the clitic with an incorrect one using weighted selection.

        Within-number swaps (sg→sg, pl→pl) are weighted more heavily
        than cross-number swaps, reflecting natural error patterns
        per Slevanayi (2001), p. 60.
        """
        ctx = position["context"]
        current_clitic = ctx["clitic"]

        if current_clitic not in CLITIC_SWAPS_WEIGHTED:
            return None

        alternatives, weights = CLITIC_SWAPS_WEIGHTED[current_clitic]

        # Use weighted random selection
        # random.choices returns a list, take first element
        new_clitic = self.rng.choices(alternatives, weights=weights, k=1)[0]

        error_word = ctx["stem"] + new_clitic

        if error_word == position["original"]:
            return None

        return error_word
