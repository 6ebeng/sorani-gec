"""
Sorani Kurdish Linguistic Constants

Single source of truth for all linguistic constants, lexicons, and rule
encodings used across the morphology, error generation, and evaluation modules.

Extracted from the monolithic agreement.py for modularity. Every constant
is tagged with its source finding (F#1 through F#256) from the analysis
of 30 Kurdish linguistics books documented in book_findings_report.md.

Key structural findings encoded here:
    F#1  — Subject-verb agreement is the primary agreement domain in Sorani
    F#8  — Formal agreement rule (Rasul 2005): controller→target feature copy
    F#64 — Grammatical vs semantic agreement distinction (Slevanayi 2001)
    F#66 — Two agreement laws: Law 1 (subject) + Law 2 (object/ergative)
    F#80 — Split ergativity: nominative in present, ergative in past
    F#94 — Nominative case unifies both laws (Amin 2016)

The agreement typology follows Slevanayi (2001) "Agreement in the Kurdish
Language" (ڕێککەوتن لە زمانی کوردیدا), which formalises two levels:

  1. NP-internal agreement (ڕێککەوتن لە ناو فریزی ناو) [F#67]:
     Determiners agree with head nouns in number (and gender in oblique
     case). Adjectives and certain pronoun sub-types (reflexive,
     interrogative, reciprocal) are INVARIANT — they never agree with
     the head noun (Slevanayi 2001, pp. 37-48) [F#79].
     No definiteness agreement exists within the NP — head noun and
     modifiers never agree in definiteness (Slevanayi 2001, pp. 47-48) [F#90].

  2. Sentence-level agreement (ڕێککەوتن لە ناو ڕستە) [F#66]:
     Two laws govern verb agreement, both unified as nominative-case
     agreement (Slevanayi 2001, p. 89; Amin 2016 [F#94]):
       Law 1 — Subject-verb [F#1]: The verb agrees with the subject in
       person and number. Applies to intransitive verbs in ALL tenses and
       transitive verbs in PRESENT/FUTURE tenses (Slevanayi 2001, p. 89).
       Law 2 — Object-verb (ergative) [F#14, F#39]: The verb agrees with
       the OBJECT in person and number. Applies to PAST transitive verbs
       only, due to Sorani Kurdish's split-ergative alignment [F#80]
       (Slevanayi 2001, pp. 60-61; Amin 2016, pp. 17-18).

Further refinements:
  - Grammatical agreement (ڕێککەوتنی ڕێزمانی) [F#64]: formal feature
    matching between controller and target (person, number, case,
    definiteness) (Slevanayi 2001, p. 32).
  - Semantic agreement (ڕێککەوتنی واتایی) [F#65]: selectional restrictions
    based on meaning — government vs concord (Slevanayi 2001, p. 32).
  - Pronoun hierarchy [F#75, F#87]: when compound subjects combine different
    persons, the most familiar person controls agreement (1st > 2nd > 3rd)
    (Slevanayi 2001, p. 68).
  - Compound noun subjects (X و Y) always force plural verb [F#88]
    (Slevanayi 2001, p. 61).
  - Interrogative pronouns (کێ, چ) never agree with verb [F#73]
    (Slevanayi 2001, p. 81).
  - Reciprocal pronouns (ئێکدوو / هەڤدوو) as objects are invariant [F#74];
    apparent feature match with verb is formal coincidence [F#93], not
    agreement (Slevanayi 2001, pp. 82-83).
  - Oblique bare nouns agree in neither person nor number with verb [F#82, F#89]
    (Slevanayi 2001, pp. 55-56).
  - Modal (هەوەس) verbs force the embedded main verb into subjunctive
    بـ form; the indicative دە- prefix is ungrammatical after a modal
    (Amin 2016, p. 45; F#113).
  - Passive clitic reassignment [F#107]: when a transitive verb is passivised,
    the former Set 1 object clitic maps to the corresponding Set 2
    subject clitic — م→م, ت→یت, ی→ێت, مان→ین, تان→ن, یان→ن
    (Amin 2016, p. 43).
  - Clitic role switching [F#22]: in present tense, Set 1 = subject; in past
    transitive, Set 1 = object. The roles flip based on tense and
    transitivity (Haji Marf 2014; Amin 2016).
  - Three clitic sets [F#9]: Set 1 (م/ت/ی/مان/تان/یان), Set 2 (م/یت/ێت/ین/ن/ن),
    Set 3 = possessive (same forms as Set 1, but never agree)
    (Amin 2016, pp. 17-18).

The heuristic edge detection is informed by the verb conjugation paradigms
in Amin (2016) "Verb Grammar of the Kurdish Language" [F#95-F#114], and the
formal verb morpheme templates in Maaruf (2010) "Phrase Structure in Kurdish",
pp. 78-89 (14 templates showing clitic ordering for simple and complex verbs).

Additional syntactic constraints come from Mukriani (2000) "Syntax of the
Kurdish Sentence" (سینتاکسی ڕستەی کوردی), which contributes:
  - Quantifier position effect [F#144]: pre-nominal quantifiers/numerals
    control plural agreement on the verb (دوو کەس هاتن), whereas
    post-nominal ordinals do not (کەسی دووەم هات) — pp. 24-26.
  - Tense-sensitive clitic position [F#149]: bound clitics move to
    pre-verbal position in negative past (نەمگرت) and past progressive
    (دەمگرت), while remaining post-verbal in simple past (گرتم) — p. 64.
  - Word order constraints [F#145, F#150]: subject is never sentence-final;
    verb never precedes subject when both are present — pp. 62, 75-77.
  - Complete grammatical agreement (گونجانیێکی ڕێزمانی تەواو) exists
    between subject and verb in the VP — p. 73 Conclusion.
"""

import logging

logger = logging.getLogger(__name__)

# Subject pronouns with person/number mapping
# Source: Amin (2016), pp. 17-18; F#9 (three clitic sets), F#83 (two pronoun sets)
SUBJECT_PRONOUNS = {
    "من": ("1", "sg"),
    "تۆ": ("2", "sg"),
    "ئەو": ("3", "sg"),
    "ئێمە": ("1", "pl"),
    "ئێوە": ("2", "pl"),
    "ئەوان": ("3", "pl"),
}

# Present-tense verb prefixes
PRESENT_VERB_PREFIXES = {"دە", "ئە"}

# All verb prefixes (including negation and mood)
ALL_VERB_PREFIXES = {"دە", "ئە", "نا", "نە", "ب", "مە"}

# Past-tense verb stems (common and irregular)
# Source: Amin (2016), pp. 15, 51-52 — includes 32 irregular stems (ناویزەکان)
#         Kurdish Academy grammar (2018), pp. 80–181
# F#11 (two verb roots), F#15 (conjugation tables), F#17 (irregular stems),
# F#25 (past morpheme inventory: ا, ی, و, ت, د)
PAST_VERB_STEMS = {
    # --- Common regular stems ---
    "کرد", "چوو", "هات", "گوت", "دیت", "نووسی", "خوێند",
    "کڕی", "فرۆشت", "برد", "خوارد", "دا", "گرت", "ناسی",
    "گەیشت", "ویست", "بوو",
    # --- Irregular stems (Amin 2016, p. 15; ناویزەکان) ---
    "دەست", "مرد", "ئاوا",
    "نارد", "شکاند", "ڕشت", "لایەند",
    "فێرکرد", "شوشت", "دۆزی", "فرۆشت",
    "هاوشت", "پشت", "بەخشی", "کوشت",
    "هەڵکشا", "کەوت", "لاچوو",
    "چنی", "کوتا", "سووتاند", "پوختکرد",
    # --- Suppletive pairs (Amin 2016, p. 37) ---
    "زانی", "توانی", "بینی",
    # --- Expanded from Kurdish Academy + Rasul (2005) ---
    "هێنا", "تاشی", "پێچا", "پاراست", "خست",
    "بەست", "بیست", "کێڵا", "پێوا",
    "سپارد", "ژمارد", "بژارد", "شارد",
    # --- Patientive intransitive stems ---
    "سووتا", "پسا", "خنکا", "ڕژا", "شکا", "دڕا",
    "بڕا", "ترسا", "فەوتا", "خزی", "تەقی",
    # --- Agentive intransitive expanded ---
    "گەڕا", "فڕی", "وەستا", "گریا", "هەستا", "پەڕی",
    "دانیشت", "نووست", "ڕۆیشت", "نیشت",
    "خەوت", "مایەوە", "خلیسکا", "ڕووخا", "شەقا",
    "سووڕا", "هەڵسا",
    # --- Causative past stems ---
    "سووتاند", "وەستاند", "خزاند", "مراند", "تەقاند",
    "ڕژاند", "کەواند", "نواند", "فراند", "بەزاند",
    # --- Additional high-frequency stems (Kurdish Academy 2018) ---
    "گێڕا", "پیاچوو", "ڕاچوو", "دەرچوو", "پێکهات",
    "پەیدابوو", "بزری", "لێدا", "پێدا", "تێکرد",
    "ئاژاند", "ڕاکرد", "ڕامابوو", "هەڵگرت", "هەڵدا",
    "پێشکەشکرد", "دابەزی", "سەرکرد", "وەرگرت",
    "پێشوازیکرد", "تەرخانکرد", "گۆڕی", "ڕاگیرا",
    "داخست", "بەربووەوە", "ڕابوو", "ئاگاداربوو",
    "قسەکرد", "کارکرد", "باسکرد", "یاریکرد",
    "دەسکرد", "گوێگرت", "پشتگرت",
}

# Intransitive past-tense verb stems (verb agrees with subject even in past)
# Source: Slevanayi (2001), pp. 60-61; Amin (2016), pp. 37, 51;
#         Rasul (2005), pp. 4–5; Kurdish Academy grammar (2018), pp. 80–106
# Split into agentive (volitional) and patientive (non-volitional/ڕوودان)
# F#13 (agentive vs patientive subtypes), F#24 (exhaustive verb lists),
# F#35 (two subtypes of intransitive)
INTRANSITIVE_PAST_STEMS_AGENTIVE = {
    "چوو", "هات", "نیشت", "گەیشت", "بوو",
    "ڕۆیشت", "مایەوە", "خەوت", "نووست",
    "گەڕا", "فڕی", "وەستا", "گریا", "دانیشت",
    "هەستا", "پەڕی", "سووڕا", "هەڵسا",
}

INTRANSITIVE_PAST_STEMS_PATIENTIVE = {
    # ڕوودان (happening) verbs — patient-like subject, no imperative
    "مرد", "کەوت",
    "خلیسکا", "ئاوابوو", "ڕووخا", "شەقا",
    "سووتا", "پسا", "خنکا", "ڕژا", "شکا", "دڕا",
    "بڕا", "ترسا", "فەوتا", "خزی", "تەقی",
    "بزی", "کۆکی", "پژمی", "ژیا", "برژا",
}

INTRANSITIVE_PAST_STEMS = (
    INTRANSITIVE_PAST_STEMS_AGENTIVE | INTRANSITIVE_PAST_STEMS_PATIENTIVE
)

# Transitive past-tense verb stems (verb agrees with OBJECT — ergative)
# Source: Slevanayi (2001), pp. 60-61; Amin (2016), pp. 37, 51;
#         Kurdish Academy grammar (2018), pp. 106–107, 144–156;
#         Rasul (2005), pp. 13–14
# F#14 (past transitive double clitic), F#39 (clitic placement rules),
# F#66 (Law 2 — object agrees with verb in past transitive)
TRANSITIVE_PAST_STEMS = {
    "کرد", "گوت", "دیت", "نووسی", "خوێند", "کڕی", "فرۆشت",
    "برد", "خوارد", "دا", "گرت", "ناسی", "ویست",
    # Additional from Amin (2016, pp. 15, 37) — more transitives
    "نارد", "شکاند", "ڕشت", "لایەند", "فێرکرد", "شوشت",
    "دۆزی", "هاوشت", "بەخشی", "کوشت", "سووتاند",
    "چنی", "کوتا", "پوختکرد", "زانی", "توانی", "بینی",
    # Expanded: Kurdish Academy + Rasul (2005)
    "هێنا", "تاشی", "پێچا", "پاراست", "خست",
    "بەست", "بیست", "کێڵا", "پێوا",
    "سپارد", "ژمارد", "بژارد", "شارد",
    # Causative transitives
    "سووتاند", "وەستاند", "خزاند", "مراند", "تەقاند",
    "ڕژاند", "کەواند", "نواند", "فراند", "بەزاند",
    "پساند", "وەراند",
    # Additional high-frequency transitives
    "لێدا", "پێدا", "تێکرد", "ئاژاند", "ڕاکرد",
    "هەڵگرت", "هەڵدا", "پێشکەشکرد", "سەرکرد", "وەرگرت",
    "تەرخانکرد", "گۆڕی", "داخست", "قسەکرد", "باسکرد",
    "کارکرد", "دەسکرد", "گوێگرت", "پشتگرت",
}

# Verbs using the ات allomorph for 3SG agreement
# Source: Amin (2016), pp. 21-22 — F#96 (3sg present allomorphy)
# F#34 (3sg zero marker in past intransitive & passive)
# F#28 (3sg agreement clitic is Ø, not ە or ێ)
# F#48 (3sg is NOT a pronoun — Author Haji Marf's key argument)
# Only 8 verbs use ات instead of ێت (present root ends in ۆ or ە)
AT_ALLOMORPH_VERBS = {
    "دەکات",     # does (کردن)
    "دەخوات",    # eats (خواردن)
    "دەخات",     # puts (خستن)
    "دەشوات",    # washes (شوشتن)
    "دەپوات",    # can/stirs
    "دەبات",     # takes (بردن)
    "دەدات",     # gives (دان)
    "دەگات",     # reaches (گەیشتن)
}

# Invariant forms: these token types NEVER agree with the head noun
# Source: Slevanayi (2001), pp. 38-48, 77-78, 81, 82-83, 87-88
# F#79 (adjective invariance confirmed), F#2 (no adj-gender in Sorani)
# Includes adjectives, reflexive pronouns, interrogative pronouns,
# reciprocal pronouns, possessive pronouns, and quantifiers/numerals
INVARIANT_ADJECTIVES = {
    "گەورە", "بچووک", "باش", "خراپ", "نوێ", "کۆن", "جوان",
    "درێژ", "کورت", "گرنگ", "ئاسان", "سەخت", "تازە", "ڕەش",
    "سپی", "سوور", "زەرد", "شین", "خۆش", "بەرز", "نزم",
    # Additional common adjectives
    "پان", "تەسک", "قووڵ", "فراوان", "تەنک", "ئەستوور",
    "سارد", "گەرم", "وشک", "تەڕ", "قورس", "سووک",
    "تاریک", "ڕووناک", "هێمن", "توند", "نەرم", "ڕەق",
    "پیر", "گەنج", "دەوڵەمەند", "هەژار", "زیرەک", "گێل",
    "لاواز", "بەهێز", "نوقم", "چەورەبن",
}
INVARIANT_PRONOUNS = {
    "خۆم", "خۆت", "خۆی", "خۆمان", "خۆتان", "خۆیان",  # reflexive
    "کێ", "چی", "کام",                                   # interrogative
    "یەکتر", "یەکدی",                                     # reciprocal
}

# Interrogative pronouns — never agree at sentence level
# Source: Slevanayi (2001), p. 81 — F#73 (completely invariant نەگۆڕ)
# When used as subject or object, no agreement edge should be built.
INTERROGATIVE_PRONOUNS = {"کێ", "چی", "چ", "کام"}

# Reciprocal pronouns — invariant, always object
# Source: Slevanayi (2001), pp. 82-83; F#74 (reciprocal invariant),
# F#93 (formal coincidence vs true agreement)
# When reciprocal is object of past transitive, verb defaults to 3sg
# because the reciprocal has no person/number features to copy.
# Any apparent feature match is formal coincidence (لێکچوونی شێوەیی).
RECIPROCAL_PRONOUNS = {"یەکتر", "یەکدی", "ئێکدی", "ئێکتر"}
# Possessive pronouns NEVER trigger agreement (Slevanayi 2001, pp. 77-78)
# F#71 (possessive pronouns NEVER agree), F#12 (agreement vs possessive sets),
# F#83 (two pronoun sets with opposite agreement behavior)
INVARIANT_POSSESSIVES = {
    "مم", "ت", "ی", "مان", "تان", "یان",  # bound possessive clitics
}
# Quantifiers/numerals always trigger PLURAL on the verb (Slevanayi 2001, pp. 87-88)
# F#77 (quantifier/numeral always plural verb)
# But they do NOT agree in number with the noun (the noun stays singular after numerals)
QUANTIFIER_FORMS = {
    "هەموو", "گشت", "هەندێک", "چەند", "تۆزێک", "هیچ",
    "زۆر", "کەم", "فرە", "یەکێک", "دوو", "سێ", "چوار",
    "پێنج", "شەش", "حەوت", "هەشت", "نۆ", "دە",
    # Higher numerals and additional quantifiers
    "یازدە", "دوازدە", "سێزدە", "چواردە", "پازدە",
    "شازدە", "حەڤدە", "هەژدە", "نۆزدە", "بیست",
    "سی", "چل", "پەنجا", "شەست", "حەفتا",
    "هەشتا", "نەوەد", "سەد", "هەزار",
    "چەندین", "چەندێک", "هەرچەند",
}

# Quantifier/Numeral Position Effect — Mukriani (2000, pp. 24-26)
# When a quantifier/numeral PRECEDES the noun (pre-nominal position),
# it controls plural agreement on the verb:
#   دوو کەس هاتن   ('two people came')  → plural verb
# When a numeral FOLLOWS the noun as an ordinal (post-nominal position),
# it does NOT affect verb agreement — the noun controls:
#   کەسی دووەم هات  ('the second person came')  → singular verb
# This positional asymmetry is critical for the agreement graph builder:
# only pre-nominal quantifiers should generate quantifier_verb edges.
QUANTIFIER_POSITION_PRENOMINAL = True  # flag for build_agreement_graph

# Mass nouns — never take plural suffix directly; need a measure word
# Source: Slevanayi (2001), pp. 46-47, 53, 57; F#68 (mass noun + measure word)
# The measure word (پێوەر) controls verb agreement, not the mass noun.
# Example: من دوو پەرداخیت شیری ڤەخوارن — verb agrees with "two glasses"
MASS_NOUNS = {
    "شیر", "گەنم", "برنج", "ئاو", "خۆڵ", "خوێن",
    "نان", "هەوا", "دارو", "نەفت", "ئاسن", "ئالتوون",
    "قوماش", "پارە", "هەناسە",
}

# Measure words (پێوەر) that quantify mass nouns
# Source: Slevanayi (2001), pp. 47, 53, 57
MEASURE_WORDS = {
    "پەرداخ", "لبا", "گلاس", "کیلۆ", "تۆن", "لیتر",
    "پارچە", "دانە", "بۆتل", "تەشت", "گۆنی", "قاپ",
}

# Collective nouns — morphologically singular but semantically plural
# Source: Slevanayi (2001), pp. 45-46, 58; F#69 (collective noun dual behavior)
# Bare (no determiner): singular verb — لەشکەر هاتØ
# With هەموو or determiner: plural verb — هەموو جووتیار هاتن
COLLECTIVE_NOUNS = {
    "لەشکەر", "گەل", "خەڵک", "ئایل", "خێزان",
    "کۆمەڵ", "جەماوەر", "سوپا", "دەستە", "پۆلیس",
    "مەڕ", "هێڵ", "جووتیار",
}

# Demonstrative pronouns — dual behavior (pro-form vs. determiner)
# Source: Slevanayi (2001), pp. 83-86; F#70 (demonstrative dual behavior)
# As pro-form (standalone): agrees directly with verb
# As determiner (in NP): agrees with head noun, NOT directly with verb
DEMONSTRATIVES = {"ئەو", "ئەم", "ئەوە", "ئەمە"}

# Existential هەبوون — three-way distinction
# Source: Slevanayi (2001), pp. 75-77; F#72 (three-way distinction)
# F#198 (possessive هەبوون paradigm), F#207 (weak verb agreement)
# 1. بوونی هاتنەئارایی (becoming) — regular intransitive, Law 1
# 2. هەبوون (existence) — agreement set agrees, both tenses
# 3. هەبوون (possession) — possessive set NEVER agrees, verb stays 3sg
EXISTENTIAL_STEMS = {"هەبوو", "هەیە", "نییە", "هەبێت", "نەبێت",
                     "هەبوون", "نەبوو", "بوو", "بوون", "هەن"}

# Vocative suffixes — agree with imperative verb in number
# Source: Slevanayi (2001), pp. 16, 72-73
# ۆ = masculine singular, ێ = feminine singular, ینۆ = plural
VOCATIVE_SUFFIXES = {
    "sg_masc": "ۆ",
    "sg_fem": "ێ",
    "pl": "ینۆ",
}

# Negation conjunction — first-conjunct agreement rule
# Source: Slevanayi (2001), pp. 61, 68; F#85 (first conjunct controls)
# With نە...نە or یا...یا, verb agrees with first conjunct only
NEGATION_CONJUNCTIONS = {"نە", "یا"}

# Coordination conjunction — used to detect compound subjects
# Source: Slevanayi (2001), p. 61; F#88 (compound noun subjects force plural)
# Compound NP subjects force plural verb
COORDINATION_CONJUNCTION = "و"

# Common proper nouns — cannot take indefinite/plural markers
# Source: Slevanayi (2001), pp. 43-44; F#86 (proper noun constraints)
# Proper nouns have unique reference and resist ەک (indefinite) and ان/ین (plural).
# *هەولێرەک, *هەولێران, *کوردستانان are all ungrammatical.
COMMON_PROPER_NOUNS = {
    "هەولێر", "سلێمانی", "کوردستان", "عێراق", "دهۆک", "کەرکووک",
    "بەغدا", "سۆران", "بادینان", "ئەربیل", "موسڵ", "ئامەد",
    "مەهاباد", "سنە", "ڕانیە", "چوارتا", "حەڵەبجە",
}

# Noun suffixes that indicate the noun is NOT bare
# Source: Slevanayi (2001), pp. 47-48, 55-56; F#82 (bare noun person-only),
# F#89 (oblique bare noun zero agreement), F#90 (no definiteness agreement)
# Bare nouns (no determiner, no plural marker) agree only in person (3rd)
# with the verb — number features are absent.
NOUN_MARKING_SUFFIXES = ("ەکە", "ەکان", "ێک", "ان", "ین", "یان", "یەکە", "انێک")

# Pronoun familiarity hierarchy for compound subject resolution
# Source: Slevanayi (2001), p. 68; F#75 (compound subject hierarchy),
# F#87 (strict word order in compound subjects)
# 1st person > 2nd person > 3rd person
PERSON_HIERARCHY = {"1": 0, "2": 1, "3": 2}

# Clitic forms (longest first for matching)
CLITIC_FORMS = ["مان", "تان", "یان", "م", "ت", "ی"]

# Clitic Set 1 vs Set 2 — essential for ergativity modeling
# Source: Slevanayi (2001), pp. 34-37; Finding #9, #22
# Set 1 (pronominal clitics م/ت/ی/مان/تان/یان):
#   Present tense → marks SUBJECT (agent)
#   Past transitive → marks OBJECT (patient)
#   Role flips by tense + transitivity.
# Set 2 (verbal agreement suffixes م/یت/ێت/ین/ن):
#   Always on the verb stem — marks subject in present,
#   marks object in past transitive (ergative alignment).
# Set 3 (possessive) shares forms with Set 1 but NEVER triggers agreement.
CLITIC_SET_1: dict[str, tuple[str, str]] = {
    "مان": ("1", "pl"),
    "تان": ("2", "pl"),
    "یان": ("3", "pl"),
    "م":   ("1", "sg"),
    "ت":   ("2", "sg"),
    "ی":   ("3", "sg"),
}

CLITIC_SET_2: dict[str, tuple[str, str]] = {
    "م":   ("1", "sg"),
    "یت":  ("2", "sg"),
    "ێت":  ("3", "sg"),
    "ات":  ("3", "sg"),
    "ین":  ("1", "pl"),
    "ن":   ("3", "pl"),
}

# Pre-head NP determiners — elements that precede the head noun WITHOUT ezafe
# Source: Farhadi (2013), pp. 15-16 — F#117 (no ezafe required)
# These attach directly before the head noun. Inserting ezafe between
# a pre-head determiner and the head is an error (*زۆری کتێب → زۆر کتێب).
# Multiple pre-head elements can stack: هەر تەنیا ئەو دوو منداڵە
PRE_HEAD_DETERMINERS = {
    # Restrictive particles
    "هەر", "تەنیا",
    # Demonstratives (also in DEMONSTRATIVES set, but listed here for clarity)
    "ئەم", "ئەو",
    # Degree words
    "زۆر", "هەندێ", "یەکجار", "کەمێک",
    # Question particle
    "چەند",
    # Negation
    "هیچ",
    # Titles (نازناو)
    "مامۆستا", "دکتۆر", "پرۆفیسۆر", "ئەندازیار", "پزیشک",
}

# Note: Numerals (دوو, سێ, چوار, ...) and superlatives (باشترین, etc.)
# are also pre-head but are already in QUANTIFIER_FORMS or detectable
# by the ترین suffix. The above set covers non-numeric pre-head items.

# Definiteness marker migration rule
# Source: Farhadi (2013), pp. 16-17 — F#115 (definiteness marker migration)
# In descriptive NPs (adj modifier): ەکە migrates to LAST modifier
#   قوتابییە زیرەکەکە (the smart student) — ەکە on adjective
# In possessive NPs: ەکە STAYS on head noun
#   خانووەکەی ئازاد (Azad's house) — ەکە on head noun
# Flag for the NP agreement checker to validate marker position.
DEFINITE_MARKER_MIGRATION_DESCRIPTIVE = True  # ەکە goes to last modifier
DEFINITE_MARKER_MIGRATION_POSSESSIVE = False  # ەکە stays on head

# Complement vs adjunct verb sets — verbs requiring obligatory complements
# Source: Farhadi (2013), pp. 31-33 — Finding #118
# These verbs REQUIRE a locative/prepositional complement; without it,
# the sentence is ungrammatical. Adjuncts can be freely omitted.
# Test: removal of complement → ungrammatical (*ئەو بوو vs ئەو لە ژوورەکەوە بوو)
COMPLEMENT_REQUIRING_VERBS = {
    "بوو",       # existential-locative: requires لە...دا complement
    "دانا",      # place: requires location
    "نا",        # put: requires location
    "دامەزراند",  # establish: requires location/institution
}

# ---------------------------------------------------------------------------
# Interrogative sentence formation — Farhadi (2013), pp. 49-51 (Finding #119)
# ---------------------------------------------------------------------------
# Four question types in Sorani Kurdish, each with specific formation rules:
# 1. Yes/no: ئایا/ئەرێ + declarative, OR intonation-only (no marker)
# 2. Wh-question: question word replaces the questioned constituent
# 3. Choice: X یان Y structure
# 4. Tag: declarative + tag particle (وانییە, وایە, باشە)

# Particles prefixed to yes/no questions for formal marking
YESNO_QUESTION_PARTICLES = {"ئایا", "ئەرێ"}

# Wh-question words (overlaps with INTERROGATIVE_PRONOUNS but also includes
# adverbial question words not in the pronoun set)
QUESTION_WORDS = {
    "کێ",     # who
    "چی",     # what
    "چ",      # what (short form)
    "کام",    # which
    "کوا",    # where
    "چۆن",    # how
    "کەی",    # when
    "چەند",   # how many/much
    "بۆچی",   # why
    "بۆ",     # why (short form, context-dependent)
}

# Tag question forms appended to declarative sentences
TAG_QUESTION_FORMS = {"وانییە", "وایە", "باشە"}

# ---------------------------------------------------------------------------
# Negative concord — Haji Marf (2014), pp. 294-295 (Finding #121)
# ---------------------------------------------------------------------------
# هیچ/چ (negative pronouns) require the verb to be negated. Using هیچ
# with a positive verb is ungrammatical: *هیچم پێ دەخورێ.
# Correct: هیچم پێ ناخورێ. Applies to extended forms too.
NEGATIVE_CONCORD_TRIGGERS = {"هیچ", "چ"}
NEGATIVE_CONCORD_EXTENDED = {
    "هیچ کەس", "هیچ شت", "هیچ شتێک", "هیچ کات",
    "هیچ کەسێک", "هیچ جۆرێک",
}
# Verb negation prefixes that satisfy negative concord
VERB_NEGATION_PREFIXES = ("نا", "نە", "مە")

# ---------------------------------------------------------------------------
# هەرگیز/هیچ tense restriction — KSA (2011), p. 251 (Finding #256)
# ---------------------------------------------------------------------------
# Negation adverbs هەرگیز, هیچ ڕەنگێ, هیچ کلۆجێک are restricted to past
# and future tenses. Present-tense copular/stative is ungrammatical:
#   ✓ هەرگیز لێرە نەبووم (past)   ✓ هەرگیز لێرە نابم (future)
#   ✗ *هەرگیز لێرە نیم (present)  ✗ *هەرگیز من دز نیم (present)
HERGIZ_ADVERBS = {"هەرگیز", "هیچ ڕەنگێ", "هیچ کلۆجێک"}
HERGIZ_BANNED_TENSES = {"present"}  # only past and future are valid

# ---------------------------------------------------------------------------
# ش/یش ordering asymmetry — Haji Marf (2014), pp. 263, 245-250 (Finding #122)
# ---------------------------------------------------------------------------
# On demonstratives: ش/یش MUST precede the clitic (ئەمیشم, ئەمەشم).
# On خۆ (reflexive): ش/یش can go EITHER side (خۆشم AND خۆمیش are OK).
YISH_BEFORE_CLITIC_ONLY = {"ئەم", "ئەمە", "ئەمان", "ئەمانە",
                           "ئەو", "ئەوە", "ئەوان", "ئەوانە"}
YISH_BIDIRECTIONAL = {"خۆ"}  # خۆشم and خۆمیش both valid

# ---------------------------------------------------------------------------
# Demonstrative+preposition contraction — Haji Marf (2014), pp. 263-264
# (Finding #123)
# ---------------------------------------------------------------------------
# بە/لە + ئەم/ئەو → contracted forms. Writing the uncontracted form
# (*بە ئەم instead of بەم) is a segmentation/spelling error.
DEMONSTRATIVE_PREPOSITION_CONTRACTIONS: dict[tuple[str, str], str] = {
    ("بە", "ئەم"): "بەم",
    ("لە", "ئەم"): "لەم",
    ("بە", "ئەو"): "بەو",
    ("لە", "ئەو"): "لەو",
    ("بە", "ئەمە"): "بەمە",
    ("لە", "ئەمە"): "لەمە",
    ("بە", "ئەمانە"): "بەمانە",
    ("لە", "ئەمانە"): "لەمانە",
    ("بە", "ئەوە"): "بەوە",
    ("لە", "ئەوە"): "لەوە",
    ("بە", "ئەوانە"): "بەوانە",
    ("لە", "ئەوانە"): "لەوانە",
}
# Lexicalized demonstrative compounds (never split)
DEMONSTRATIVE_COMPOUNDS = {"ئەمسال", "ئەمڕۆ", "ئەمشەو"}

# ---------------------------------------------------------------------------
# Possessive pronoun هی/ئی — no clitics allowed
# Haji Marf (2014), pp. 291-293 (Finding #124)
# ---------------------------------------------------------------------------
# هی/ئی cannot take clitics or ش/یش. *هیم, *هیتان are ungrammatical.
# Correct: هی من, هی تۆ, هی ئێمە.
CLITIC_BARRED_PRONOUNS = {"هی", "ئی", "هین"}

# ---------------------------------------------------------------------------
# Imperative mood clitic restrictions — Haji Marf (2014), p. 192
# (Finding #125)
# ---------------------------------------------------------------------------
# Intransitive imperatives (normal or explicit negation) NEVER take clitics.
# Transitive imperatives: clitic between بـ/مە and verb root.
# Permission/non-explicit negation: clitic at end for both.
IMPERATIVE_MOOD_PREFIXES = {"ب", "مە"}

# ---------------------------------------------------------------------------
# Compound verb (noun+verb) clitic insertion — Haji Marf (2014), pp. 217-218
# (Finding #126)
# ---------------------------------------------------------------------------
# In compound verbs, Set 1 clitic inserts between nominal component and
# verbal root: سەر + مان + شکاند → سەرمانشکاند.
# These nominal components are common first elements of compound verbs.
COMPOUND_VERB_NOMINAL_ELEMENTS: set[str] = {
    "سەر",    # سەرشکاندن (to defeat)
    "نان",    # نانخواردن (to eat food / to dine)
    "ڕێگا",   # ڕێگاگرتن (to block the road)
    "بڵند",   # بڵندکردن (to raise)
    "فڕێ",    # فڕێدان (to throw)
    "جێگە",   # جێگەکردنەوە (to replace)
    "دەس",    # دەسکردن (to start)
    "پشت",    # پشتگرتن (to support)
    "گوێ",    # گوێگرتن (to listen)
    "باس",    # باسکردن (to discuss)
    "کار",    # کارکردن (to work)
    "قسە",    # قسەکردن (to speak)
    "یاری",   # یاریکردن (to play)
}

# ---------------------------------------------------------------------------
# Causative suffix clitic position — Haji Marf (2014), pp. 215-216
# (Finding #127)
# ---------------------------------------------------------------------------
# -ەوە/-وە suffix: clitic BEFORE suffix (کردمانەوە).
# -اندن suffix: clitic AFTER suffix (گەیاندمان).
CAUSATIVE_SUFFIX_EWE = {"ەوە", "وە"}     # clitic goes BEFORE these
CAUSATIVE_SUFFIX_ANDN = {"اندن", "اند"}  # clitic goes AFTER these

# ---------------------------------------------------------------------------
# Reciprocal pronoun یەکتر — plural subject requirement
# Haji Marf (2014), pp. 296-297 (Finding #128)
# ---------------------------------------------------------------------------
# یەکتر requires plural subject. Takes ی-ezafe, Set 1 clitics, ش/یش.
# Dialectal variants: یەکدی, یەکدوو (southern), هەڤ, هەڤدو (Kurmanci).
RECIPROCAL_VARIANTS: set[str] = {
    "یەکتر", "یەکدی", "یەکدوو",  # Sorani variants
    "هەڤ", "هەڤدو",               # Kurmanci variants
}
# Reciprocal pronouns ALWAYS require plural subjects
RECIPROCAL_REQUIRES_PLURAL = True

# ---------------------------------------------------------------------------
# Clitic phrase-level attachment: first-element rule
# Fatah & Qadir (2006), pp. 33-34 (Finding #129)
# ---------------------------------------------------------------------------
# Clitics (Set A) attach to the FIRST element in the clause/phrase (object,
# aspect marker, negation, preposition, degree word), NOT to the verb root.
# Grammatical prefixes (دە-, نا-, نە-) always bind to the verb root.
CLITIC_HOST_CATEGORIES: list[str] = [
    "object",         # پارەکەم برد
    "aspect_marker",  # دەم کڕی
    "negation",       # نەم دەکڕی
    "degree_adverb",  # زۆرم گرت
    "preposition",    # لێم سەند
    "verb_final",     # بردم (when verb is the first/only element)
]
# Grammatical prefixes that never "float" — always on verb root
VERB_BOUND_PREFIXES: set[str] = {"دە", "نا", "نە"}

# ---------------------------------------------------------------------------
# Morpheme collision tolerance: identical stacking is grammatical
# Fatah & Qadir (2006), pp. 44-46 (Finding #130)
# ---------------------------------------------------------------------------
# When morpheme boundaries produce identical adjacent sequences, Kurdish
# tolerates the stacking. Each instance serves a different function.
# These patterns must NOT be flagged as typos or repetition errors.
PERMITTED_MORPHEME_DOUBLINGS: dict[str, str] = {
    "مم":     "stem-final م + 1sg clitic م (e.g. گەنمم)",
    "مانمان": "stem-final مان + 1pl possessive (e.g. نیشتمانمان)",
    "تانتان": "stem-final تان + 2pl possessive (e.g. کورتانتان)",
    "یانیان": "stem-final یان + 3pl possessive (e.g. گریانیان)",
    "انان":   "stem-final ان + plural ان (e.g. دانان)",
    "ەکەکە":  "definiteness ەکە + relative-clause کە (e.g. تورەکەکە)",
}

# ---------------------------------------------------------------------------
# Derivational-before-grammatical affix ordering (universal)
# Fatah & Qadir (2006), p. 57 (Finding #131)
# ---------------------------------------------------------------------------
# In ALL Sorani word categories: base + derivational + grammatical.
# Grammatical affixes are ALWAYS outer (further from root). This is inviolable.
DERIV_BEFORE_GRAM_RULE_INVIOLABLE = True
# Common derivational suffixes (must be inner / closer to root)
DERIVATIONAL_SUFFIXES: set[str] = {
    "ەتی", "مەند", "گەر", "دار", "بەند", "وانە", "ەوان", "ەوار",
    "ین", "ۆڵە", "یلە", "یلکە", "وولە", "انە",
}
# Common grammatical suffixes (must be outer / further from root)
GRAMMATICAL_SUFFIXES: set[str] = {
    "ەکە", "کە", "یەک", "ێک", "ان", "یان",
    "تر", "ترین", "ە", "ی",
}

# ---------------------------------------------------------------------------
# Preverb transitivity flip: clitic set must change
# Fatah & Qadir (2006), pp. 72-73 (Finding #132)
# ---------------------------------------------------------------------------
# هەڵ (and others) can reverse verb transitivity. When a root changes from
# transitive to intransitive (or vice versa), the clitic set assignment per
# F#55 must correspond to the NEW transitivity, not the original root's.
PREVERB_TRANSITIVITY_FLIPS: dict[str, dict[str, str]] = {
    # preverb: {root: new_transitivity}
    "هەڵ": {
        "کرد": "intransitive",   # کرد(tr) → هەڵکرد(intr: got up)
        "کوتا": "intransitive",  # کوتا(tr) → هەڵکوتا(intr: set out)
        "دا": "intransitive",    # دا(tr) → هەڵدا(intr: rose/swelled)
    },
}

# ---------------------------------------------------------------------------
# Fatah & Qadir (2006), p. 42 (Finding #133)
# ---------------------------------------------------------------------------
# Same-set clitic exclusion: members of the same clitic set (A or B) cannot
# co-occur in a simple sentence.  Only cross-set (A+B) combinations are valid.
# One exception: past transitive with reflexive خۆ.
SAME_SET_CLITIC_EXCLUSION_IN_SIMPLE = True
SAME_SET_EXCLUSION_EXCEPTION_XO_PAST_TRANSITIVE = True

# ---------------------------------------------------------------------------
# Fatah & Qadir (2006), pp. 55-56 (Finding #134)
# ---------------------------------------------------------------------------
# Definiteness marker ەکە can NEVER attach to a conjugated verb form.
# Valid hosts: nouns, adjectives, nominalized infinitives.
# Allomorph: ەکە after consonant, کە after vowel.
# EXCEPTION (F#231, Ibrahim 1988 pp. 22-24): و/وو-final nouns take ەکە
# (consonant pattern) despite ending in a vowel.
DEFINITENESS_VERB_ATTACHMENT_BAN = True
DEFINITENESS_VALID_ON_NOMINALIZED_INFINITIVES = True
DEFINITENESS_ALLOMORPHS: dict[str, str] = {
    "post_consonant": "ەکە",
    "post_vowel": "کە",
    "post_u_uu": "ەکە",  # و/وو-final: patterns with consonant, not vowel
}
INDEFINITE_ALLOMORPHS: dict[str, str] = {
    "post_consonant": "ێک",
    "post_vowel": "یەک",
    "post_u_uu": "ێک",  # و/وو-final: ێک not *یەک (Ibrahim 1988 p. 23)
}

# ---------------------------------------------------------------------------
# Fatah & Qadir (2006), p. 71 (Finding #135)
# ---------------------------------------------------------------------------
# The surface morpheme ە has 5 distinct grammatical functions.
# Disambiguation depends on syntactic context.
MORPHEME_E_FUNCTIONS: dict[str, str] = {
    "present_tense": "هێناومە — present-relevance / perfect aspect marker",
    "demonstrative_postbound": "ئەو کوڕە — part of circumfix ئەو...ە",
    "definiteness_informal": "کوڕە هات — informal definite reference",
    "ezafe_replacement": "کوڕە کورد = کوڕی کورد — replaces ی-ezafe",
    "masculine_vocative": "کوڕە وا مەکە — masculine address marker",
}

# ---------------------------------------------------------------------------
# Fatah & Qadir (2006), p. 52 (Finding #136)
# ---------------------------------------------------------------------------
# Definiteness ەکە always precedes ALL other grammatical suffixes.
# گوڵ+ەکە+ان = گوڵەکان ✓   *گوڵ+ان+ەکە = *گوڵانەکە ✗
# Extends F#131 by specifying internal ordering within grammatical suffixes.
DEFINITENESS_PRECEDES_ALL_GRAM_SUFFIXES = True
GRAMMATICAL_SUFFIX_ORDER: list[str] = ["ەکە", "ان"]  # leftmost = closest to stem

# ---------------------------------------------------------------------------
# Sa'id (2009), p. 77 (Finding #137)
# ---------------------------------------------------------------------------
# Systematic preposition-to-case-role mapping.  Using the wrong preposition
# for a given case role is a detectable GEC error.
PREPOSITION_CASE_MAP: dict[str, str | None] = {
    "Agent":       "لە لایەن",   # passive agent marker (+ ەوە)
    "Object":      None,          # direct object is bare (no preposition)
    "Experiencer": "بۆ",          # also بە in some frames
    "Instrument":  "بە",
    "Source":       "لە",          # + وە (ablative)
    "Location":    "لە",          # also بە
    "Time":        "لە",          # also بە
    "Benefactive": "بە",          # also بۆ
    "Goal":        "بەرەو",
    "Comitative":  "لەگەڵ",
    "Path":        "بەدرێژایی",
}

# ---------------------------------------------------------------------------
# Abbas & Sabir (2020), pp. 23-26 (Finding #217)
# Citing Muhammad (2003:47) and Abdullah (1993:51-64)
# ---------------------------------------------------------------------------
# Complete closed-class Sorani preposition inventory.
SORANI_SIMPLE_PREPOSITIONS: frozenset[str] = frozenset({
    "لە", "بە", "بۆ", "بێ", "تا", "هەتا", "تاکو", "تاوەکو", "هەتاوەکو",
    "و", "ی", "ە", "ش", "یش", "لەگەڵ", "لەتەک",
})

SORANI_NOMINAL_PREPOSITIONS: frozenset[str] = frozenset({
    "پێش", "پاش", "تەنیشت", "بەرامبەر",
})

SORANI_COMPOUND_PREPOSITIONS: frozenset[str] = frozenset({
    "بەبێ", "لەبۆ", "لە لایەن", "بەدرێژایی", "بەرەو",
})

# ---------------------------------------------------------------------------
# Abbas & Sabir (2020), pp. 23-24, 27 (Finding #218)
# Qadir (2002), pp. 62-66 (Finding #250) — circumposition pattern:
#   لە+NP+دا (locative), لە+NP+ەوە (ablative), بە+NP+دا (intermediary),
#   بە+NP+ەوە (instrumental).  Postposition is OBLIGATORY; dropping it = error.
# ---------------------------------------------------------------------------
# Preposition لە has 7+ semantic functions.  Each function pairs with a
# specific bound suffix.  Missing or wrong suffix = detectable GEC error.
LE_POLYSEMY: dict[str, dict[str, str]] = {
    "locative":    {"suffix": "دا", "example": "لە دهۆکدا"},
    "partitive":   {"suffix": "",   "example": "لە نانەکە"},
    "part_of_set": {"suffix": "",   "example": "لە هەموو کچەکان"},
    "comparative": {"suffix": "",   "example": "لە شوان زیرەکترە"},
    "temporal":    {"suffix": "دا", "example": "لە دووشەمەدا"},
    "ablative":    {"suffix": "وە", "example": "لە بازارەوە"},
    "abstract":    {"suffix": "دا", "example": "لە کێشەدایە"},
}

# ---------------------------------------------------------------------------
# Abbas & Sabir (2020), pp. 28-29 (Finding #221)
# ---------------------------------------------------------------------------
# Tolerance rule: certain preposition pairs are interchangeable in shared
# semantic domains.  The GEC system should NOT flag these as errors.
PREPOSITION_SUBSTITUTABLE_PAIRS: frozenset[frozenset[str]] = frozenset({
    frozenset({"بۆ", "بە"}),     # benefactive / experiencer overlap
    frozenset({"لە", "بە"}),     # temporal / locative overlap in some frames
    frozenset({"بۆ", "بەرەو"}),  # goal / destination overlap (F#222)
})

# ---------------------------------------------------------------------------
# Abbas & Sabir (2020), pp. 24, 27 (Finding #222)
# ---------------------------------------------------------------------------
# Preposition بۆ has 4 semantic functions.
BO_POLYSEMY: dict[str, dict[str, str]] = {
    "goal":        {"suffix": "",  "example": "بۆ سەرچنار"},
    "benefactive": {"suffix": "",  "example": "بۆ منداڵەکانم"},
    "temporal":    {"suffix": "",  "example": "بۆ بەیانی"},
    "duration":    {"suffix": "",  "example": "بۆ سێ مانگ"},
}

# ---------------------------------------------------------------------------
# Abbas & Sabir (2020), p. 24 (Finding #224)
# ---------------------------------------------------------------------------
# Preposition بە combines with postbound suffixes (parallel to لە patterns).
BA_POSTBOUND: dict[str, dict[str, str]] = {
    "intermediary": {"suffix": "دا", "example": "بە براکەمدا"},
    "instrumental": {"suffix": "وە", "example": "بە پێوە"},
}

# ---------------------------------------------------------------------------
# Abbas & Sabir (2020), pp. 24-25 (Finding #223)
# ---------------------------------------------------------------------------
# Prepositions outside Sa'id's 12-role case system.
PRIVATIVE_PREPOSITIONS: frozenset[str] = frozenset({"بێ", "بەبێ"})
LIMITATIVE_PREPOSITIONS: frozenset[str] = frozenset({
    "تا", "هەتا", "تاکو", "تاوەکو", "هەتاوەکو",
})

# ---------------------------------------------------------------------------
# Abbas & Sabir (2020), pp. 25-26 (Finding #225)
# ---------------------------------------------------------------------------
# Nominal prepositions have temporal/spatial duality.
# Spatial function requires لە...ەوە frame; temporal function is bare.
NOMINAL_PREP_SPATIAL_FRAME: dict[str, dict[str, str]] = {
    "پێش": {"spatial": "لە پێش ... ەوە", "temporal": "پێش"},
    "پاش": {"spatial": "لە پاش ... ەوە", "temporal": "پاش"},
}

# ---------------------------------------------------------------------------
# Ibrahim (1988), pp. 5-14 (Finding #226)
# ---------------------------------------------------------------------------
# Free-standing vocative particles with gender constraints.
# Distinct from bound vocative suffixes (F#76, F#135, F#147).
VOCATIVE_PARTICLES: dict[str, str] = {
    "ئەی": "neutral",   # gender-neutral, general use
    "هۆ": "masculine",   # masculine addressee only
    "هێ": "feminine",    # feminine addressee only (adds ی to name)
    "یا": "sacred",      # Arabic loan, for sacred/revered names
    "ۆ": "masculine",    # masculine only, Kurmanci-heavy
}

# ---------------------------------------------------------------------------
# Ibrahim (1988), pp. 66-67 (Finding #227)
# ---------------------------------------------------------------------------
# Emphatic superlative prefix هەرە stacks with تر and ترین.
# All three stacking patterns are grammatical (not double-marking errors).
EMPHATIC_SUPERLATIVE_PREFIX: str = "هەرە"
COMPARATIVE_SUFFIX: str = "تر"
SUPERLATIVE_SUFFIX: str = "ترین"

# ---------------------------------------------------------------------------
# Ibrahim (1988), pp. 117-118 (Finding #228)
# ---------------------------------------------------------------------------
# Purpose subordinators — freely interchangeable in purpose clauses.
# تا also functions as duration subordinator (not shared by others).
PURPOSE_SUBORDINATORS: frozenset[str] = frozenset({
    "بۆئەوەی", "تا", "تاوەکوو", "هەتاکوو",
})
DURATION_SUBORDINATORS: frozenset[str] = frozenset({
    "تا", "هەتا", "هەتاکوو",
})

# ---------------------------------------------------------------------------
# Ibrahim (1988), pp. 92-95 (Finding #229)
# ---------------------------------------------------------------------------
# بەڵکوو (corrective conjunction) requires a correlative in the first clause.
# تەنیا → first clause verb must be NEGATIVE.
# نەوەکوو → first clause verb must be POSITIVE.
BALKUU_CORRELATIVES: dict[str, str] = {
    "تەنیا": "negative",    # first clause verb must be negative
    "نەوەکوو": "positive",  # first clause verb must be positive
}

# ---------------------------------------------------------------------------
# Ibrahim (1988), pp. 75-142 (Finding #230)
# ---------------------------------------------------------------------------
# Complete conjunction inventory with positional constraints.
SORANI_COORDINATING_CONJUNCTIONS: frozenset[str] = frozenset({
    "و", "ش", "بەڵام", "یا", "ئەوسا", "ئەوجا",
    "بەڵکوو", "وەکوو", "ئینجا", "ئەگینا", "وە", "ئێستا",
})
SORANI_SUBORDINATING_CONJUNCTIONS: frozenset[str] = frozenset({
    "کە", "بۆئەوەی", "کەچی", "تا", "هەتا", "هەتاکوو",
    "چونکە", "لەبەرئەوەی کە", "ئەگەر", "گەر", "مەگەر",
    "ئەگەرچی", "بۆیە",
})
# Positional constraint: ئەگەر always clause-initial; کەچی always medial.
CLAUSE_INITIAL_ONLY_CONJUNCTIONS: frozenset[str] = frozenset({
    "ئەگەر", "گەر", "مەگەر", "ئەگەرچی",
})
MEDIAL_ONLY_CONJUNCTIONS: frozenset[str] = frozenset({"کەچی"})

# ---------------------------------------------------------------------------
# Ibrahim (1988), pp. 26-27 (Finding #232)
# ---------------------------------------------------------------------------
# When two nouns are conjoined by و, the plural marker ان attaches to the
# LAST noun only.  The first noun remains bare.
# e.g. کوڕ و کچان (not *کوڕان و کچان)
COORDINATED_NP_PLURAL_LAST_ONLY: bool = True

# ---------------------------------------------------------------------------
# Ibrahim (1988), pp. 82-84 (Finding #233)
# ---------------------------------------------------------------------------
# ش has two distinct functions depending on context:
# 1) As sole connector → coordinating conjunction (each clause needs its own subject)
# 2) With و also present → repetition/additive marker (= "also"), no constraint
SH_CONJUNCTION_REQUIRES_DISTINCT_SUBJECTS: bool = True

# ---------------------------------------------------------------------------
# Ibrahim (1988), pp. 14-18 (Finding #234)
# ---------------------------------------------------------------------------
# Interrogative چ forces indefinite marker (ێک/یەک) on the following noun.
# *چ کتێب and *چ کتێبەکە are both ungrammatical; must be چ کتێبێک.
INTERROGATIVE_CH_FORCES_INDEFINITE: bool = True

# ---------------------------------------------------------------------------
# Ibrahim (1988), pp. 35-36 (Finding #235)
# ---------------------------------------------------------------------------
# Four sentence-initial optative particles (all functionally equivalent).
# These precede the clause and pair with optative verb morphology (F#158).
OPTATIVE_SENTENCE_PARTICLES: frozenset[str] = frozenset({
    "خۆزگە", "خۆزیا", "بریا", "کاشکی",
})

# ---------------------------------------------------------------------------
# Shwani (2003), pp. 70-71, Conclusion #16 (Finding #236)
# ---------------------------------------------------------------------------
# When لەگەڵ links two nouns as subject, the verb agrees ONLY with the
# first noun (grammatical subject).  The second noun is a complement.
# Contrast with و where both nouns force plural (F#88).
# لەگەڵ can appear at most once per sentence.
COMITATIVE_PREPOSITION: str = "لەگەڵ"
COMITATIVE_SINGULAR_AGREEMENT: bool = True  # verb agrees with 1st noun only

# ---------------------------------------------------------------------------
# Shwani (2003), pp. 87-88, Conclusion #20 (Finding #237)
# ---------------------------------------------------------------------------
# Verb tenses in compound-sentence clauses joined by coordinating
# conjunctions must match or be closely related.
# Exception: ئەنجا/ئینجا ("then") permits sequential tenses.
SEQUENTIAL_CONJUNCTIONS: frozenset[str] = frozenset({
    "ئەنجا", "ئینجا", "ئەوجا",
})

# ---------------------------------------------------------------------------
# Shwani (2003), pp. 89-91, Conclusion #23 (Finding #238)
# ---------------------------------------------------------------------------
# Paired conjunctions: position before subject → 2 subjects (one per clause);
# position after subject → 1 shared subject.
PAIRED_CONJUNCTIONS: tuple[tuple[str, str], ...] = (
    ("هەم", "هەم"),
    ("نە", "نە"),
    ("یان", "یان"),
    ("یا", "یا"),
    ("چ", "چ"),
)

# ---------------------------------------------------------------------------
# Shwani (2003), pp. 87-100, Conclusion #21 (Finding #239)
# ---------------------------------------------------------------------------
# In compound sentences, identical elements (subject / object / verb) in the
# second clause are obligatorily elided.
COMPOUND_ELLIPSIS_OBLIGATORY: bool = True

# ---------------------------------------------------------------------------
# Shwani (2003), pp. 78-80, Conclusion #18 (Finding #240)
# ---------------------------------------------------------------------------
# بۆ becomes interrogative "why" when it appears sentence-finally with an
# intransitive verb.  Otherwise it is a preposition "for/to".
BO_INTERROGATIVE_CONTEXT: str = "intransitive_final"

# ---------------------------------------------------------------------------
# Shwani (2003), pp. 41-65, Conclusions #1,4,5,6,35 (Finding #241)
# ---------------------------------------------------------------------------
# Three-way connective classification; dual-function instruments.
# و, ی, ە serve as both prepositions AND conjunctions.
# تا serves as both preposition AND subordinator.
# ش/یش serves as both preposition AND conjunction.
DUAL_FUNCTION_CONNECTIVES: frozenset[str] = frozenset({
    "و", "ی", "ە", "تا", "ش", "یش",
})
# Subordinator overlap: temporal کە can replace conditional ئەگەر.
SUBORDINATOR_OVERLAP: dict[str, str] = {
    "کە": "ئەگەر",  # temporal → conditional
}

# ---------------------------------------------------------------------------
# Shwani (2003), pp. 45-47 (Finding #242, deep pass)
# ---------------------------------------------------------------------------
# Preposition postfix allomorphy: the postfix form (-دا/-یا/-وە) after
# بۆ, بە, لە depends on whether the preceding noun ends in a vowel or
# consonant.  After vowels, the epenthetic ی variant (-یا) is available.
# بۆ+وە is restricted to locative adverbs only.
PREPOSITION_POSTFIX_ALLOMORPHS: dict[str, dict[str, list[str]]] = {
    "consonant_final": {"locative": ["دا", "ا"], "directional": ["وە"]},
    "vowel_final": {"locative": ["دا", "یا"], "directional": ["وە"]},
}
BO_POSTFIX_WE_LOCATIVE_ONLY: bool = True  # بۆ+وە restricted to locatives

# Preposition-to-clitic transformation: بە→پێ, لە→{لێ,تێ}, بۆ→ە
# Corrected per F#245: بە→پێ (not تێ); لە→لێ (general) and لە→تێ
# (penetration/involvement semantics).
PREPOSITION_CLITIC_FORMS: dict[str, list[str]] = {
    "بە": ["پێ"],
    "لە": ["لێ", "تێ"],
    "بۆ": ["ە"],
}

# ---------------------------------------------------------------------------
# Shwani (2003), pp. 79-80 (Finding #243, deep pass)
# ---------------------------------------------------------------------------
# ش/یش attachment position determines semantic scope (4 rules).
# Position 1: Subject+ش+V → co-participation
# Position 2: Subject+VerbPrefix+ش+V → emphasis
# Position 3: N₁+ش+N₂+V → surprise/incredulity
# Position 4: N₁+ش+و+N₂+ش+V → dual-state binding
SHISH_POSITION_COUNT: int = 4  # four distinct attachment positions

# ---------------------------------------------------------------------------
# Shwani (2003), p. 71 (Finding #244, deep pass)
# ---------------------------------------------------------------------------
# When نە appears WITHIN a و-coordinated subject (نە A و نە B),
# و still governs → verb must be PLURAL.
# Contrast: paired نە...نە without و → first-conjunct singular (F#85).
NEGATION_WITHIN_WA_COORDINATION_PLURAL: bool = True

# ---------------------------------------------------------------------------
# [Author unknown] (~2018), pp. 13-14, 20-21 (Finding #245)
# ---------------------------------------------------------------------------
# Complete preposition-to-bound-form transformation table with diagnostics.
# بە → {پێ (dative/benefactive), وە (dialectal/temporary), ی (imperative)}
# لە → {لێ (general), تێ (penetration/involvement)}
# بۆ → {ە (directional after verbs)}
# Diagnostic: substitute full preposition + independent pronoun to identify
# source (e.g. لێت → لە تۆ, پێی → بە ئەو, تێمان → لە ئەومان).
PREPOSITION_BOUND_FORMS: dict[str, dict[str, str]] = {
    "بە": {"پێ": "dative_benefactive", "وە": "dialectal_temporary", "ی": "imperative_dative"},
    "لە": {"لێ": "general", "تێ": "penetration_involvement"},
    "بۆ": {"ە": "directional"},
}

# ---------------------------------------------------------------------------
# [Author unknown] (~2018), pp. 21-22 (Finding #246)
# ---------------------------------------------------------------------------
# ش/یش conjunction vs. marker: removability diagnostic.
# removable (no meaning change) → grammatical marker (emphasis)
# sole connective in compound sentence → conjunction (non-removable)
# co-occurs with و → both are conjunctions
# When conjunction, F#237 tense concordance and F#239 ellipsis apply.
SHISH_REMOVABLE_IS_MARKER: bool = True
SHISH_WITH_WA_BOTH_CONJUNCTION: bool = True

# ---------------------------------------------------------------------------
# [Author unknown] (~2018), p. 11 (Finding #247, deep pass)
# ---------------------------------------------------------------------------
# Preposition postfix scoping in coordinated NPs: the postfix (-دا, -وە)
# attaches to the LAST conjunct only.  The preposition scopes over all.
# Correct:  لە جەژن و خۆشی و تاڵی و شاییدا
# Wrong:   *لە جەژندا و لە خۆشیدا و لە تاڵیدا و لە شاییدا
PREPOSITION_POSTFIX_LAST_CONJUNCT_ONLY: bool = True

# ---------------------------------------------------------------------------
# [Author unknown] (~2018), p. 19 (Finding #248, deep pass)
# ---------------------------------------------------------------------------
# Demonstrative circumfix (ئەم...ە / ئەو...ە) inherently marks definiteness.
# The definite article ەکە/ەکان CANNOT co-occur on the same noun head.
# Grammatical markers (ان plural, etc.) CAN nest inside the circumfix.
# Demonstratives NEVER attach to pronouns.
DEMONSTRATIVE_BLOCKS_DEFINITE_ARTICLE: bool = True
DEMONSTRATIVE_BLOCKS_PRONOUN: bool = True
# ---------------------------------------------------------------------------
# Duration preposition بۆ is optional: both bare and marked forms are valid.
DURATION_PREPOSITION_OPTIONAL: bool = True

# ---------------------------------------------------------------------------
# Sa'id (2009), p. 73 (Finding #138)
# ---------------------------------------------------------------------------
# Object selection constraints: only these case roles can become grammatical
# object.  Agent can NEVER become object.
OBJECTIVISABLE_ROLES: frozenset[str] = frozenset({"Object", "Benefactive", "Location"})
NON_SUBJECTIVISABLE_ROLES: frozenset[str] = frozenset({"Source", "Goal", "Time"})

# ---------------------------------------------------------------------------
# Sa'id (2009), p. 69 (Finding #139)
# ---------------------------------------------------------------------------
# Extended 7-role subject selection hierarchy.  The verb agrees with the
# highest-ranked role present in the clause.
# Differs from F#6's 5-role hierarchy; here Object outranks Experiencer and
# Instrument, and Location / Time are explicitly included.
SUBJECT_SELECTION_HIERARCHY: list[str] = [
    "Agent",       # highest priority
    "Object",
    "Experiencer",
    "Instrument",
    "Location",
    "Time",
    "Benefactive", # lowest priority
]

# ---------------------------------------------------------------------------
# Sa'id (2009), p. 75 (Finding #140)
# ---------------------------------------------------------------------------
# Passive subject promotion restriction.  Only these three case roles can be
# promoted to grammatical *subject* in a passive sentence.  Crucially,
# Instrument is EXCLUDED even though it CAN be an active subject (per F#5).
# These three passive-eligible roles never co-occur in one sentence.
PASSIVE_SUBJECTIVISABLE_ROLES: frozenset[str] = frozenset({
    "Object",       # باس — e.g. دەرگاکە کرایەوە
    "Experiencer",  # کارتێکراو — e.g. ئەو بریندار کرا
    "Benefactive",  # سوودمەند — e.g. بۆ قوتابییەکە وانەکە نووسرا
})

# ---------------------------------------------------------------------------
# Sa'id (2009), pp. 89-91 (Finding #141)
# ---------------------------------------------------------------------------
# Relative clause کە deletion rules.
# • Restrictive (بەستراو): کە may be deleted when head noun is definite
#   (ـەکە or ئەم/ئەو), but ezafe ی must then appear on the head noun.
# • Non-restrictive (بەڕەڵا): کە is OBLIGATORY and cannot be deleted.
# • کە works uniformly across all case roles (شوێن, کات, ئامێر, etc.).
RELATIVE_CLAUSE_MARKER: str = "کە"

# ── F#143: Definiteness marker distribution in coordinated vs. ezafe NPs ──
# (Mukriyani 2000, pp. 15-16)
# In و-coordinated NPs, definiteness marker can attach to last-only or all.
# In ezafe-linked NPs, constituent order is FIXED (head-first); reordering is ✗.
COORDINATION_CONJUNCTION: str = "و"

# ── F#144: Quantifier position controls verb agreement ──
# (Mukriyani 2000, pp. 24-26)
# Pre-subject quantifier → forces plural verb.
# Post-subject ordinal (ـەم/ـەمین) → NO effect on verb agreement.
ORDINAL_SUFFIXES: tuple[str, ...] = ("ەم", "ەمین", "ەمیین")

# ── F#145: Direct object must precede verb (hard constraint) ──
# (Mukriyani 2000, pp. 73-74)
# بەرکاری ڕاستەوخۆ ناکەوێتە کۆتایی ڕستەوە.
# Indirect object (بەرکاری تیان) CAN follow verb.
DIRECT_OBJECT_PRECEDES_VERB: bool = True

# ── F#146: Intransitive complement–بە alternation ──
# (Mukriyani 2000, pp. 38-40)
# Complement of intransitive verb can move post-verb ONLY with بە.
# *سالار بوو پاڵەوانی گوند (without بە) is ungrammatical.
INTRANSITIVE_COMPLEMENT_PREP: str = "بە"

# ── F#147: Vocative marker targets common nouns only ──
# (Mukriyani 2000, pp. 19-20, §1-18)
# When both common noun and proper name are present in vocative,
# the vocative suffix attaches ONLY to the common noun.
# کوڕە سیروان بخوێنە ✓, *سیروانە بخوێنە ✗
VOCATIVE_SUFFIXES: dict[str, str] = {
    "masculine": "ە",
    "feminine": "ێ",
    "plural": "ینە",
}

# ── F#149: Conditional بـ as sole root–clitic intervenor ──
# (Mukriyani 2000, pp. 64-65, §3-8)
# Across all tenses, no morpheme except conditional بـ can split
# the root–clitic bond.  *گرتدم, *نووسینم are always ungrammatical.
CONDITIONAL_CLITIC_INTERVENOR: str = "ب"

# ── F#150: V-S inversion restricted to emphatic copular sentences ──
# (Mukriyani 2000, pp. 21-22, §1-19)
# Subject always precedes verb. V-before-S is only valid with
# copular/nominal predicates for emphasis (pride, optimism, pessimism).
# With lexical verbs, V-S is NEVER grammatical.
SUBJECT_PRECEDES_VERB: bool = True

# ── F#151: Conjoined NP + post-modifier scope ambiguity ──
# (Farhadi 2013, pp. 19-20, §گرێی ناوی ئاڵۆز)
# "N₁ و N₂ ی modifier" has two valid parses: wide scope (modifier
# applies to both nouns) and narrow scope (modifier applies only to
# the last noun).  Both readings are grammatical — the GEC system
# must tolerate agreement variation with conjoined-NP + modifier.
CONJOINED_NP_MODIFIER_AMBIGUITY: bool = True

# ── F#152: Indirect object = adverbial — IO non-promotability ──
# (Farhadi 2013, pp. 29-30, §بەرکاری ناڕاستەوخۆ و ئاوەڵگوزارە)
# IO (PP introduced by بە/لە/بۆ) shares 8 syntactic properties with
# adverbials and CANNOT be promoted to passive subject.  Only the
# direct object (no preposition) promotes.
IO_PROMOTES_TO_PASSIVE_SUBJECT: bool = False

# ── F#153: Adverbial positional freedom — non-error rule ──
# (Farhadi 2013, pp. 27-28, §ئاوەڵگوزارە)
# Adverbials are "کەرەسەیەکی بزێوە" (movable): any sentence
# position is valid.  The system must NOT flag adverbial position
# variation as a word-order error.
ADVERBIAL_POSITION_FREE: bool = True

# ── F#154: Commissive verbs — first-person subject only ──
# (Farhadi 2013, pp. 54-55, §ڕستەی ڕاپەڕاندن)
# Commissive speech acts require 1st-person subject (من/ئێمە).
# Closed set of commissive verbs listed below.
COMMISSIVE_VERBS: tuple[str, ...] = (
    "بەڵێندان",
    "پەیماندان",
    "گفتدان",
    "ناونان",
    "سوێنددان",
    "سوێندخواردن",
)

# ── F#155: Dialectal participle morpheme variation ──
# (Rasul 2005, p. 21)
# The past-participle morpheme has three dialectal allomorphs:
#   Central (standard): و  (سوتاوە, مردوە, هاتوە)
#   Northern:           ی  (سۆتیە, مریە, هاتیە)
#   Southern:           گ  (سوزیاگە, مردگە, هاتگە)
# In Standard Sorani (Central), only و is correct.  Northern ی
# and Southern گ forms in formal writing are dialectal errors.
PARTICIPLE_MORPHEME_STANDARD: str = "و"
PARTICIPLE_MORPHEME_DIALECTAL: tuple[str, ...] = ("ی", "گ")

# ── F#253: Germian dialect agreement morpheme order reversal (Wali 2023, pp. 167-170) ──
# Extends F#191.  In past tense morphosyntactic forms:
#   Standard (Sulaimani): V + SubjAgr + ObjAgr  (e.g. نووسیـتـن)
#   Germian:              V + ObjAgr + SubjAgr  (e.g. بردیانم — reversed)
# In non-past: both dialects are IDENTICAL (no order difference).
# Exception verb وێ (want): object agreement goes at end instead of subject
#   (من تۆم دەوێت → دەمەوێیت — reversed from all other non-past verbs).
# Germian also uses only one morpheme set {م, مان} for BOTH roles in past tense,
# whereas standard Sulaimani uses {م, مان} vs {م, ین} for different roles.
GERMIAN_MORPHEME_ORDER_REVERSED: bool = True  # past-tense only
WANT_VERB_MORPHEME_EXCEPTION: str = "وێ"  # وێ/ویستن reverses non-past order

# ── F#254: Asymmetric tense sequencing in و-coordinate clauses (Maaruf 2009, pp. 84-85) ──
# In clauses coordinated with و:
#   Same tense both clauses → always OK
#   Past → Non-past → OK (temporal sequence: earlier → later)
#   Non-past → Past → UNGRAMMATICAL (*ئەو نان دەخوات و پۆشت)
#   Exception: Non-past → Past Perfect is OK (bridges to present)
# With temporal adverbs (دوێنێ، ئەمڕۆ): tense mismatch is allowed.
# Contrastive conjunctions بەڵام / کەچی: allow free tense mixing.
COORD_TENSE_SEQ_BLOCKED: tuple[str, str] = ("non-past", "past")  # only this direction is blocked
COORD_TENSE_EXEMPTING_CONJ: tuple[str, ...] = ("بەڵام", "کەچی")  # no tense-harmony required

# ── F#255: مەگەر requires subjunctive; ئەگەر accepts both moods (Maaruf 2009, pp. 121-122) ──
# مەگەر + subjunctive: ✓ مەگەر نەتگرم / بتگرم
# مەگەر + indicative: ✗ *مەگەر یاریمەدەکەم
# ئەگەر + subjunctive: ✓ ئەگەر بە پێ بڕۆین
# ئەگەر + indicative: ✓ ئەگەر ئێوە بۆ سەیران چوون
# مەگەر clause cannot be reordered; ئەگەر clause can.
MEGER_MOOD: str = "subjunctive"  # مەگەر takes subjunctive ONLY
EGER_MOOD: tuple[str, ...] = ("subjunctive", "indicative")  # ئەگەر accepts both
# (Rasul 2005, pp. 25-26)
# Adversative compound sentences require at least ONE connector
# from a matched pair: opening + closing.
ADVERSATIVE_OPENING: tuple[str, ...] = ("ئەگەرچی", "هەرچەندە", "لەگەڵئەوەشداکە")
ADVERSATIVE_CLOSING: tuple[str, ...] = ("بەڵام", "کەچی")

# ── F#157: Three-mood tense-morpheme mapping ──
# (Rasul 2005, pp. 38-41)
# Same template بـ+Root+morph+agr produces three moods depending
# on tense morpheme: ە→imperative, ێ→subjunctive, ا→optative.
MOOD_TENSE_MAP: dict[str, str] = {
    "imperative": "ە",    # present morpheme — 2nd person only
    "subjunctive": "ێ",   # future morpheme — all 6 persons
    "optative": "ا",      # past morpheme — counterfactual
}
# Negation prefix also differs by mood:
MOOD_NEG_PREFIX: dict[str, str] = {
    "imperative": "مە",
    "subjunctive": "نە",
    "optative": "نە",
}

# ── F#158: Optative paradigm template ──
# (Rasul 2005, pp. 40-41)
# Optative = Root + Past_morph + بـ + ا + Agreement.
# Extended form: + ایە (counterfactual reaching to present).
# Three dialectal surface variants (only first is standard):
#   1. Simple:   چوبام      (Root+Past+بـ+ا+Agr)
#   2. بـ-prefix: بچوام     (بـ+Root+Past+ا+Agr)
#   3. Epenthetic: بچویام   (بـ+Root+ی+Past+ا+Agr)
OPTATIVE_EXTENDED_SUFFIX: str = "ایە"

# ── F#159: Patient Participle (و) vs. Agent Participle (ر) — Tense Restriction ──
# و-participle (patient/passive) is restricted to PAST tense only.
# ر-participle (agent/active) is unrestricted — valid in both past and non-past.
# Using a و-participle form in a present/future frame is a tense-agreement error.
PATIENT_PARTICIPLE_MORPHEME: str = "و"
AGENT_PARTICIPLE_MORPHEME: str = "ر"
PATIENT_PARTICIPLE_TENSE_RESTRICTION: str = "past_only"

# ── F#160: Verb Class Dual-Membership (Agentive ↔ Patientive Intransitive) ──
# These intransitive verbs can function as BOTH agentive (بکەری) and patientive
# (بەرکاری) depending on context. Imperative is valid only in agentive reading.
DUAL_CLASS_INTRANSITIVE_VERBS: tuple[str, ...] = (
    "گریان", "ژیان", "کشان",
)

# ── F#161: Seven Speech Act Types — Punctuation Mapping ──
# Maps each speech act type to its standard sentence-final punctuation.
SPEECH_ACT_PUNCTUATION: dict[str, str] = {
    "زانیاری": ".",    # Information/News → period
    "پرسیاری": "؟",    # Interrogative → question mark
    "فەرماندان": ".",   # Imperative → period
    "داخوازی": ".",     # Subjunctive request → period
    "خۆزگەیی": ".",     # Optative → period (or !)
    "هەستدەربڕین": "!", # Emotional expression → exclamation
    "هەستورووژاندن": ".",  # Emotional arousal → period
}

# ── F#162: Polite Imperative Obligatory Preparatory Phrases ──
# Strong imperative: minimal sentence, no preparatory phrase needed.
# Polite imperative: ALWAYS requires one of these preparatory phrases at start.
POLITE_IMPERATIVE_MARKERS: tuple[str, ...] = (
    "فەرموو", "فەرمون", "تکایە", "بێ زەحمەت",
    "بە یارمه\u200cتیت", "ببوورن", "ببوورە", "بەڕێزتان",
)

# ── F#163: Allophonic Orthographic Pairs ح↔هـ, ع↔هـ, خ↔غ ──
# (Rasul 2004, pp. 127-131)
# ح and ع are NOT native Kurdish phonemes — both are allophones of هـ,
# appearing only in Arabic loanwords.  غ is an allophone of خ.
# In colloquial speech these are all neutralised.  In formal writing,
# loanwords retain the Arabic spelling, creating common spelling errors.
ALLOPHONIC_PAIRS: dict[str, str] = {
    "ح": "هـ",   # حوکم → هوکم in speech
    "ع": "هـ",   # عەلی → هەلی in speech
    "غ": "خ",    # غار / خار — no meaning distinction in Sorani
}
# These letters appear ONLY in Arabic-origin words
ARABIC_ONLY_GRAPHEMES: frozenset[str] = frozenset({"ح", "ع", "غ"})

# ── F#164: و/وو Short-vs-Long Vowel Orthographic Confusion ──
# (Rasul 2004, pp. 124-126)
# و has two functions: 1) vowel (short و vs long وو), 2) past morpheme.
# When verb root ends in -و and past morpheme -و is added → double وو.
# This causes frequent spelling errors: short و written as وو and vice versa.
SHORT_LONG_VOWEL_PAIRS: dict[str, str] = {
    "و": "وو",  # short → long (e.g. پوول vs پووڵ)
    "ی": "یی",  # short → long (see F#165)
}
PAST_MORPHEME_U: str = "و"  # triggers double-وو when root ends in و

# ── F#165: Extended ی/یی Six-Scenario Rules ──
# (Rasul 2004, pp. 126-127)
# Six scenarios produce double or triple ی:
#  1. Indefinite noun + definite ی → noun+یی  (ماڵ+ی→ماڵیی)
#  2. ی-final noun + ezafe ی → noun+یی  (دەریایی گەورەکە)
#  3. ی-final definite noun + possessive → triple-ی  (دەریایـیـیەتی)
#  4. Definite + ezafe → noun+ی+ی  (پیاوەکەیی خۆش)
#  5. Past intransitive verb + ی morpheme → verb+یی  (چوو+ی→چوویی)
#  6. ی-final root + past ی → double-یی  (هاتی+ی→هاتیی)
YI_DOUBLE_SCENARIOS: tuple[str, ...] = (
    "indefinite_plus_definite",      # Scenario 1
    "yi_final_plus_ezafe",           # Scenario 2
    "yi_final_definite_possessive",  # Scenario 3 (triple ی)
    "definite_plus_ezafe",           # Scenario 4
    "past_intransitive_yi",          # Scenario 5
    "yi_final_root_past",            # Scenario 6
)

# ── F#166: Parts of Speech Deletability Hierarchy ──
# (Rasul 2005, pp. 14-16)
# Sentence constituents ranked by deletability (most deletable first):
#   ئاوەڵکات (temporal adverb) > ئاوەڵکار (manner adverb) >
#   تەواوکار (complement) > بەرکار (object) > بکەر (subject) > لێکەر (verb)
# ئاوەڵکات ≠ ئاوەڵکار: former = time adverbial, latter = manner adverbial.
CONSTITUENT_DELETABILITY_HIERARCHY: tuple[str, ...] = (
    "ئاوەڵکات",   # temporal adverbial (most deletable)
    "ئاوەڵشوێن",  # locative adverbial
    "ئاوەڵکار",   # manner adverbial
    "تەواوکار",   # complement
    "بەرکار",     # object
    "بکەر",       # subject
    "لێکەر",      # verb/predicate (least deletable)
)

# ── F#252: Object pro-drop constraint (Wali 2023, pp. 161-163) ──
# Kurdish allows subject pro-drop but NOT independent object pro-drop.
# Object can only be dropped when subject is simultaneously dropped
# (combined subject-object drop in morphosyntax).
# Exception: verbs with inferable objects (خواردنەوە, خوێندنەوە).
# *دزەکە ∅ ڕفاندی is UNGRAMMATICAL (object dropped, subject retained).
# Kurmanjî (Northern) is NOT pro-drop at all — even subjects cannot be dropped.
PRO_DROP_OBJECT_EXCEPTION_VERBS: frozenset[str] = frozenset({
    "خواردنەوە",   # drink — object inferable
    "خوێندنەوە",   # read — object inferable
})

# ── F#167: Epenthetic ت — Non-Morphemic Vowel-Boundary Insertion ──
# (Rasul 2005, Art. 1 pp. 28-29; Art. 2 pp. 54-57)
# ت is inserted at morpheme boundaries between specific vowel
# sequences. It has NO grammatical function — purely phonological.
# Three trigger environments:
#   1. ا + ە  →  ا+ت+ە  (ئەکاتە, ئەداتە, ئەباتە, ئەڕواتە, ئەخواتە)
#   2. ێ + ەوە →  ێ+ت+ەوە (بسوڕێتەوە, نەگەڕێتەوە, ئەکاتەوە)
#   3. ێ + ە  →  ێ+ت+ە  (ئەبێتە)
# The so-called "ات allomorph" is actually ا + epenthetic ت, NOT a pronoun.
EPENTHETIC_T_ENVIRONMENTS: tuple[tuple[str, str], ...] = (
    ("ا", "ە"),    # e.g., ئەکا+ە → ئەکاتە
    ("ا", "ەوە"),  # e.g., ئەکا+ەوە → ئەکاتەوە
    ("ێ", "ەوە"),  # e.g., بسوڕێ+ەوە → بسوڕێتەوە
    ("ێ", "ە"),    # e.g., ئەبێ+ە → ئەبێتە
    ("ۆ", "ە"),    # e.g., ئەڕۆ+ە → ئەڕۆتە (ۆ-ending stems)
    ("ۆ", "ەوە"),  # e.g., ئەڕۆ+ەوە → ئەڕۆتەوە
)

# Verb stems that trigger epenthetic ت in 3sg non-past forms
# (stems ending in long vowel ا after the present-tense root)
EPENTHETIC_T_VERB_STEMS: frozenset[str] = frozenset({
    "خوا",   # خواردن → ئەخواتە
    "دا",    # دان → ئەداتە
    "کا",    # کردن/کەردن → ئەکاتە
    "با",    # بردن/بەردن → ئەباتە
    "ڕوا",   # ڕۆیشتن → ئەڕواتە
    "گا",    # گەیشتن → ئەگاتە
    "شوا",   # شوشتن → ئەشواتە
    "خا",    # خستن → ئەخاتە
    "هاژوا",  # هاژۆشتن → ئەهاژواتە
    "ڕۆ",    # ڕۆیشتن → ئەڕۆتە (ۆ-ending stem)
})

# ── F#168: خواردن Exception to Causative ا→ێ Alternation ──
# (Rasul 2005, Art. 1 p. 25)
# The systematic past→non-past infix alternation (ا→ێ) in causatives
# does NOT apply to خواردن because its original form was خوەردن.
# Regular: سوتاندم→ئەسوتێنم, شکاندت→ئەشکێنی, ناردی→ئەنێرێ
# Exception: خواردم→ئەخوام (NOT *ئەخوێنم)
CAUSATIVE_A_TO_E_MAP: dict[str, str] = {
    "ا": "ێ",  # past infix ا → non-past infix ێ
}

CAUSATIVE_A_TO_E_EXCEPTIONS: frozenset[str] = frozenset({
    "خواردن",  # original form خوەردن — ا is NOT the past morpheme
})

# Regular -اردن verbs that DO follow the ا→ێ rule
CAUSATIVE_ARDN_VERBS: frozenset[str] = frozenset({
    "سپاردن",   # سپاردم → ئەسپێرم (entrust)
    "ژماردن",   # ژماردت → ئەژمێری (count)
    "ناردن",    # ناردی → ئەنێرێ (send)
    "بژاردن",   # بژاردمان → ئەبژێرین (choose)
    "شاردن",    # شاردتانەوە → ئەشێرنەوە (hide)
})

# ── F#169: ە+ە→ا Phonological Assimilation (وێکچوون) ──
# (Rasul 2005, Art. 2 pp. 50-51)
# When two ە vowels meet at a morpheme boundary, they coalesce into ا.
# Three environments:
#   (a) ئەمە + ئەچین → ئەماچین  (1pl pronoun + present prefix)
#   (b) نە + ئەچین → ناچین    (negation + present prefix)
#   (c) ئەوە + ئەچین → ئەواچین  (demonstrative + present prefix)
# This explains WHY نا- is the present-tense negation: نە + ئە → نا.
VOWEL_COALESCENCE_RULES: dict[str, str] = {
    "ە+ە": "ا",   # two short ە vowels → long ا
}

# Morphemes whose final ە triggers the ە+ە→ا rule before ئە- prefix
COALESCENCE_TRIGGER_MORPHEMES: frozenset[str] = frozenset({
    "نە",    # negation prefix → نا- before present verbs
    "ئەمە",  # 1pl pronoun → ئەما- before present verbs
    "ئەوە",  # demonstrative → ئەوا- before present verbs
})


# ===== F#170: Compound verb morpheme slot patterns =====
# Key generalization: in past tense, Person(subject) precedes root;
# in present tense, Person(object) precedes root.
# The preverbal element(s) always anchor to the leftmost slot.
COMPOUND_VERB_PREVERBAL_ELEMENTS: dict[str, list[str]] = {
    "morphosyntactic_preposition": ["پێ", "تێ", "لێ"],
    "adverb_prefix": ["هەڵ", "دا", "ڕا", "ڕۆ", "دەر", "بەر"],
    "noun_compound_heads": ["دەست", "چاو", "سەر", "پشت", "گوێ"],
    "adjective_compound_heads": ["سور", "کز", "خراپ", "باش"],
}

# ===== F#171: Default constituent ordering =====
# Subject + Time_Adverb + Place_Adverb + Direct_Object + Indirect_Object + Verb
DEFAULT_CONSTITUENT_ORDER: list[str] = [
    "subject",
    "time_adverb",
    "place_adverb",
    "direct_object",
    "indirect_object",
    "verb",
]

# ===== F#172: Attributive ـی blocks internal determiners with proper nouns =====
# When ـی connects a common noun head to a proper noun, no determiner (ـەکە, ـێک)
# can appear inside the attributive NP on the head noun.
# *شارەکەی هەولێر  ✗   (definite marker on head + proper noun attribution)
# ئەو شارەی هەولێر ✓   (external demonstrative OK)
ATTRIBUTIVE_EZAFE_PROPER_NOUN_RULE: bool = True  # flag for error checking

# ===== F#173: Superlative ترین forces pre-nominal position =====
SUPERLATIVE_SUFFIX: str = "ترین"
SUPERLATIVE_POSITION: str = "pre-nominal"  # must precede noun, no ezafe

# ===== F#174: ـەوە postposition required by source/direction verbs =====
SOURCE_DIRECTION_VERBS: frozenset[str] = frozenset({
    "هاتن", "گەڕانەوە", "ڕۆیشتنەوە", "گەیشتنەوە",
    "دەرچوون", "هاتنەوە",
})
SOURCE_POSTPOSITION: str = "ـەوە"  # required on لە-PP for source semantics

# ===== F#176: Title nouns — proper-noun-only specifiers =====
TITLE_NOUNS: frozenset[str] = frozenset({
    "دکتۆر", "مامۆستا", "حاجی", "شەهید", "کاک", "خان", "مەلا",
    "سەرتیپ", "پاشا",
})

# ===== F#178: Possessive ـی is obligatory between noun and possessor =====
# *دەست من → must be دەستی من / دەستم
# Without ـی (or clitic contraction), N + possessor is ungrammatical.
POSSESSIVE_EZAFE_OBLIGATORY: bool = True  # flag for error checking

# ===== F#179: Possessor must be [+definite] (pronoun or proper noun) =====
# *دەستی کوڕ → ungrammatical (bare common noun as possessor)
# *دەستی منەکە → ungrammatical (determiner on pronoun possessor)
POSSESSOR_REQUIRES_DEFINITENESS: bool = True

# ===== F#180: Number and quantifier are mutually exclusive as NP specifiers =====
# *هەندێک دوو کوڕ → ungrammatical
QUANTIFIERS: frozenset[str] = frozenset({
    "هەندێک", "تۆزێک", "کەمێک", "هەموو", "گشت", "هیچ", "چەندین",
})

# ===== F#181: PP is inseparable — preposition and complement both required =====
# *کراسەکەم لە کڕی (missing complement)
# *کراسەکەم ئازاد کڕی (missing preposition)
PP_INSEPARABLE: bool = True

# ===== F#182: Attributive adjective cannot carry determiners =====
# *چاوی شینەکە → ungrammatical (determiner on attributive adjective)
# Determiners must attach to the noun head, not the adjective.
ATTRIBUTIVE_ADJ_BLOCKS_DETERMINERS: bool = True

# ===== F#183: Bare common noun cannot function as NP (subject/object) =====
# *کوڕ چوو بۆ بازار → ungrammatical (bare noun without determination)
# Must have determiner (ەکە/ێک), quantifier, stress-generic, or plural.
BARE_NOUN_REQUIRES_DETERMINATION: bool = True

# ===== F#184: Abstract nouns resist determiners =====
# *خۆشەویستییەکی → ungrammatical (*abstract + indefinite)
# *خۆشەویستییەکە → ungrammatical (*abstract + definite)
# Exception: [+distinction] context only (خۆشەویستییەکی ئەوە تایبەتە).
ABSTRACT_NOUN_RESISTS_DETERMINERS: bool = True

# ===== F#185: Demonstrative cannot modify proper noun =====
# *ئەم ئازادە → ungrammatical (demonstrative + proper noun)
# Proper nouns carry [+known] inherently, clashing with demonstrative [+identification].
DEMONSTRATIVE_BLOCKS_PROPER_NOUN: bool = True

# ===== F#186: Determiner allomorphs — phonologically conditioned =====
# Definite: ەکە (after C), یەکە (after V), کە (contracted)
# Indefinite: ێک (after C), یەک (after V or free-standing)
DETERMINER_ALLOMORPHS: dict[str, tuple[str, ...]] = {
    "definite": ("ەکە", "یەکە", "کە"),
    "indefinite": ("ێک", "یەک"),
    "definite_plural": ("ەکان", "یەکان", "کان"),
    "indefinite_plural": ("ان",),
}


# ===========================================================================
# FINDINGS TRACEABILITY INDEX
# ===========================================================================
# Maps every finding from book_findings_report.md (F#1–F#256) to its
# implementation status in the codebase. Categories:
#   ENCODED  = data structure / constant / algorithm step exists in code
#   DOCBLOCK = documented finding block exists (F#143–F#186 zone)
#   THESIS   = referenced in thesis chapter(s) but not directly in code
#   PLANNED  = GEC-relevant but not yet encoded (future work)
#   N/A      = not GEC-relevant (descriptive, Kurmanci-only, typological)
#
# This index is automatically kept in sync with code changes. When adding
# a new data structure, update the corresponding entry here.
# ===========================================================================
FINDINGS_INDEX: dict[int, dict[str, str]] = {
    # --- Book 1: Rasul (2005) "Agreement in Kurdish" ---
    1:   {"status": "ENCODED", "module": "agreement.py", "desc": "Subject-verb agreement primary domain", "ref": "docstring, build_agreement_graph"},
    2:   {"status": "ENCODED", "module": "agreement.py", "desc": "No adj-gender in Sorani (Kurmanci has)", "ref": "INVARIANT_ADJECTIVES comment"},
    3:   {"status": "THESIS",  "module": "",             "desc": "Case roles (Agent, Patient, etc.)", "ref": "Ch4 §4.3"},
    4:   {"status": "THESIS",  "module": "",             "desc": "Middle/causative voice distinction", "ref": "Ch4 §4.3"},
    5:   {"status": "THESIS",  "module": "",             "desc": "Agent/instrument disambiguation", "ref": "Ch4 §4.3"},
    6:   {"status": "THESIS",  "module": "",             "desc": "5-role subject selection hierarchy", "ref": "Ch4 §4.3"},
    7:   {"status": "THESIS",  "module": "",             "desc": "Verb class by argument frame", "ref": "Ch4 §4.3"},
    8:   {"status": "ENCODED", "module": "agreement.py", "desc": "Formal agreement rule: controller→target", "ref": "docstring"},
    9:   {"status": "ENCODED", "module": "agreement.py", "desc": "Three clitic sets (Set 1/2/3)", "ref": "SUBJECT_PRONOUNS, CLITIC_FORMS"},
    10:  {"status": "ENCODED", "module": "noun_adjective.py", "desc": "Ezafe connects head + modifier", "ref": "docstring, error strategies"},
    11:  {"status": "ENCODED", "module": "agreement.py", "desc": "Two verb roots (present/past)", "ref": "PAST_VERB_STEMS comment"},
    12:  {"status": "ENCODED", "module": "agreement.py", "desc": "Agreement vs possessive pronoun sets", "ref": "INVARIANT_POSSESSIVES comment"},
    13:  {"status": "ENCODED", "module": "agreement.py", "desc": "Agentive vs patientive intransitive", "ref": "INTRANSITIVE_PAST_STEMS comment"},
    14:  {"status": "ENCODED", "module": "agreement.py, subject_verb.py", "desc": "Past transitive double clitic", "ref": "TRANSITIVE_PAST_STEMS, build_agreement_graph"},
    15:  {"status": "ENCODED", "module": "agreement.py", "desc": "Conjugation tables (6-person paradigm)", "ref": "PAST_VERB_STEMS comment"},
    16:  {"status": "ENCODED", "module": "agreement.py", "desc": "Causative formation (- اندن)", "ref": "CAUSATIVE_SUFFIX_ANDN"},
    17:  {"status": "ENCODED", "module": "agreement.py", "desc": "Irregular verb stems", "ref": "PAST_VERB_STEMS set"},
    18:  {"status": "THESIS",  "module": "",             "desc": "Sentence types (declarative, interrogative, etc.)", "ref": "Ch4 §4.4"},
    19:  {"status": "THESIS",  "module": "",             "desc": "One present root per verb", "ref": "Ch4 §4.3"},
    20:  {"status": "ENCODED", "module": "clitic.py",    "desc": "Clitic sets named (Strong/Weak)", "ref": "docstring SET_TERMINOLOGY"},
    21:  {"status": "THESIS",  "module": "",             "desc": "Possessive never on verb root", "ref": "Ch4 §4.5"},
    22:  {"status": "ENCODED", "module": "agreement.py, clitic.py", "desc": "Clitic role switching by tense", "ref": "docstring, build_agreement_graph"},
    23:  {"status": "THESIS",  "module": "",             "desc": "Sentence hierarchy (clause structure)", "ref": "Ch4 §4.4"},
    24:  {"status": "ENCODED", "module": "agreement.py", "desc": "Exhaustive intransitive verb lists", "ref": "INTRANSITIVE_PAST_STEMS comment"},
    25:  {"status": "ENCODED", "module": "agreement.py", "desc": "Past morpheme inventory (ا,ی,و,ت,د)", "ref": "PAST_VERB_STEMS comment"},
    26:  {"status": "DOCBLOCK", "module": "agreement.py", "desc": "Phonological reduction in fast speech", "ref": "F#169 coalescence"},
    27:  {"status": "ENCODED", "module": "agreement.py", "desc": "Double ی scenarios", "ref": "YI_DOUBLE_SCENARIOS (F#165)"},
    28:  {"status": "ENCODED", "module": "agreement.py", "desc": "3SG zero clitic (Ø, not ە or ێ)", "ref": "AT_ALLOMORPH_VERBS comment"},
    29:  {"status": "ENCODED", "module": "tense_agreement.py", "desc": "Ergativity detail (morphologically incomplete)", "ref": "docstring"},
    30:  {"status": "THESIS",  "module": "",             "desc": "Morphosyntax formulas (Maaruf)", "ref": "Ch4 §4.5, Ch5 §5.3"},
    31:  {"status": "ENCODED", "module": "agreement.py", "desc": "Pro-drop (subject droppable)", "ref": "F#252 PRO_DROP_OBJECT_EXCEPTION_VERBS"},
    32:  {"status": "THESIS",  "module": "",             "desc": "Verb argument structure", "ref": "Ch4 §4.3"},
    33:  {"status": "THESIS",  "module": "",             "desc": "Preposition-clitic fusion", "ref": "Ch4 §4.5"},
    34:  {"status": "ENCODED", "module": "agreement.py", "desc": "3SG zero marker in past intrans/passive", "ref": "AT_ALLOMORPH_VERBS comment"},
    35:  {"status": "ENCODED", "module": "agreement.py, subject_verb.py, tense_agreement.py", "desc": "Two subtypes of intransitive (agentive/patientive)", "ref": "INTRANSITIVE_PAST_STEMS split, RUUDAN_PRESENT_STEMS"},
    36:  {"status": "ENCODED", "module": "agreement.py", "desc": "Causative formation complete", "ref": "CAUSATIVE_SUFFIX_EWE, CAUSATIVE_A_TO_E_MAP"},
    37:  {"status": "ENCODED", "module": "tense_agreement.py", "desc": "Passive morpheme (را/ران)", "ref": "docstring F#56, F#102, F#107"},
    38:  {"status": "ENCODED", "module": "agreement.py", "desc": "8 tense system", "ref": "MOOD_TENSE_MAP, F#98 in tense_agreement.py"},
    39:  {"status": "ENCODED", "module": "agreement.py", "desc": "Past transitive clitic placement rules", "ref": "TRANSITIVE_PAST_STEMS comment"},
    40:  {"status": "ENCODED", "module": "tense_agreement.py", "desc": "Present perfect ە marker", "ref": "docstring F#40"},
    41:  {"status": "THESIS",  "module": "",             "desc": "Non-past formation rules", "ref": "Ch4 §4.3"},
    42:  {"status": "ENCODED", "module": "agreement.py", "desc": "Imperative formation (ب/مە prefix)", "ref": "IMPERATIVE_MOOD_PREFIXES"},
    43:  {"status": "ENCODED", "module": "agreement.py", "desc": "Negation patterns (نا/نە/مە)", "ref": "VERB_NEGATION_PREFIXES, MOOD_NEG_PREFIX"},
    44:  {"status": "ENCODED", "module": "subject_verb.py", "desc": "Intransitive/transitive stem pairs", "ref": "SUPPLETIVE_CAUSATIVE_PAIRS"},
    45:  {"status": "ENCODED", "module": "clitic.py",    "desc": "10 pronoun categories (Haji Marf)", "ref": "docstring"},
    46:  {"status": "ENCODED", "module": "clitic.py",    "desc": "Clitic position by tense/structure", "ref": "docstring 5 rules, F#99"},
    47:  {"status": "ENCODED", "module": "clitic.py",    "desc": "Set 2 on non-verbal hosts", "ref": "docstring Rule 5"},
    48:  {"status": "ENCODED", "module": "agreement.py", "desc": "3SG is NOT a pronoun (Haji Marf)", "ref": "AT_ALLOMORPH_VERBS comment"},
    49:  {"status": "ENCODED", "module": "noun_adjective.py", "desc": "Ezafe and pronoun interactions", "ref": "docstring F#49"},
    50:  {"status": "ENCODED", "module": "clitic.py",    "desc": "Set 2 on prepositions/adjectives", "ref": "docstring F#50"},
    51:  {"status": "N/A",     "module": "",             "desc": "Kurmanci gender differences (not Sorani)", "ref": ""},
    52:  {"status": "THESIS",  "module": "",             "desc": "Double clitic in ditransitive", "ref": "Ch4 §4.5"},
    53:  {"status": "THESIS",  "module": "",             "desc": "Word class criteria (POS tests)", "ref": "Ch4 §4.2"},
    54:  {"status": "THESIS",  "module": "",             "desc": "VP structure analysis", "ref": "Ch4 §4.4"},
    55:  {"status": "ENCODED", "module": "clitic.py",    "desc": "Clitic set determined by transitivity+tense", "ref": "docstring SET_TERMINOLOGY"},
    56:  {"status": "ENCODED", "module": "tense_agreement.py", "desc": "Passive morpheme (ر+ا decomposition)", "ref": "docstring F#56"},
    57:  {"status": "ENCODED", "module": "agreement.py", "desc": "SOV word order", "ref": "DEFAULT_CONSTITUENT_ORDER"},
    58:  {"status": "THESIS",  "module": "",             "desc": "Verb valency patterns", "ref": "Ch4 §4.3"},
    59:  {"status": "ENCODED", "module": "noun_adjective.py, analyzer.py", "desc": "Ezafe deletion error", "ref": "error strategy A"},
    60:  {"status": "ENCODED", "module": "noun_adjective.py, analyzer.py", "desc": "Ezafe allomorphs (ی / بـ)", "ref": "error strategy B"},
    61:  {"status": "ENCODED", "module": "agreement.py", "desc": "Imperative markers (ب/مە)", "ref": "IMPERATIVE_MOOD_PREFIXES"},
    62:  {"status": "ENCODED", "module": "agreement.py", "desc": "Negation prefixes (نا/نە/مە)", "ref": "VERB_NEGATION_PREFIXES"},
    63:  {"status": "N/A",     "module": "",             "desc": "Tool noun derivation (not GEC)", "ref": ""},
    # --- Book 10: Slevanayi (2001) "Agreement in Kurdish" ---
    64:  {"status": "ENCODED", "module": "agreement.py", "desc": "Grammatical vs semantic agreement", "ref": "docstring"},
    65:  {"status": "ENCODED", "module": "agreement.py", "desc": "Government vs concord distinction", "ref": "docstring F#65"},
    66:  {"status": "ENCODED", "module": "agreement.py", "desc": "Two agreement laws (Law 1 + Law 2)", "ref": "docstring, build_agreement_graph"},
    67:  {"status": "ENCODED", "module": "agreement.py", "desc": "NP-internal agreement", "ref": "build_agreement_graph Step 4"},
    68:  {"status": "ENCODED", "module": "agreement.py", "desc": "Mass noun + measure word agreement", "ref": "MASS_NOUNS"},
    69:  {"status": "ENCODED", "module": "agreement.py", "desc": "Collective noun dual behavior", "ref": "COLLECTIVE_NOUNS"},
    70:  {"status": "ENCODED", "module": "agreement.py", "desc": "Demonstrative dual behavior", "ref": "DEMONSTRATIVES"},
    71:  {"status": "ENCODED", "module": "agreement.py", "desc": "Possessive pronouns NEVER agree", "ref": "INVARIANT_POSSESSIVES"},
    72:  {"status": "ENCODED", "module": "agreement.py", "desc": "Existential هەبوون three-way", "ref": "EXISTENTIAL_STEMS, build_agreement_graph Step 6"},
    73:  {"status": "ENCODED", "module": "agreement.py", "desc": "Interrogative pronouns fully invariant", "ref": "INTERROGATIVE_PRONOUNS, Step 2b"},
    74:  {"status": "ENCODED", "module": "agreement.py", "desc": "Reciprocal pronouns invariant", "ref": "RECIPROCAL_PRONOUNS"},
    75:  {"status": "ENCODED", "module": "agreement.py", "desc": "Compound subject person hierarchy", "ref": "PERSON_HIERARCHY, _resolve_compound_subject_person"},
    76:  {"status": "ENCODED", "module": "agreement.py", "desc": "Vocative-imperative number agreement", "ref": "VOCATIVE_SUFFIXES, Step 7"},
    77:  {"status": "ENCODED", "module": "agreement.py", "desc": "Quantifier/numeral → plural verb", "ref": "QUANTIFIER_FORMS"},
    78:  {"status": "N/A",     "module": "",             "desc": "Semantic anomaly typology (not GEC)", "ref": ""},
    79:  {"status": "ENCODED", "module": "agreement.py", "desc": "Adjective invariance confirmed", "ref": "INVARIANT_ADJECTIVES"},
    80:  {"status": "ENCODED", "module": "agreement.py, tense_agreement.py", "desc": "Split ergativity", "ref": "docstring, build_agreement_graph"},
    81:  {"status": "N/A",     "module": "",             "desc": "Kurmanci case system (not Sorani)", "ref": ""},
    82:  {"status": "ENCODED", "module": "agreement.py", "desc": "Bare noun person-only agreement", "ref": "NOUN_MARKING_SUFFIXES, _is_bare_noun, Step 2c"},
    83:  {"status": "ENCODED", "module": "agreement.py", "desc": "Two pronoun sets (agreement vs possessive)", "ref": "SUBJECT_PRONOUNS, INVARIANT_POSSESSIVES"},
    84:  {"status": "N/A",     "module": "",             "desc": "Kurmanci ergative case marking", "ref": ""},
    85:  {"status": "ENCODED", "module": "agreement.py, subject_verb.py", "desc": "First conjunct agreement (نە...نە / یا...یا)", "ref": "NEGATION_CONJUNCTIONS, Step 2"},
    86:  {"status": "ENCODED", "module": "agreement.py", "desc": "Proper noun constraints", "ref": "COMMON_PROPER_NOUNS"},
    87:  {"status": "ENCODED", "module": "agreement.py", "desc": "Familiarity hierarchy (1st>2nd>3rd)", "ref": "PERSON_HIERARCHY"},
    88:  {"status": "ENCODED", "module": "agreement.py, subject_verb.py", "desc": "Compound noun subjects force plural", "ref": "COORDINATION_CONJUNCTION, Step 2a"},
    89:  {"status": "ENCODED", "module": "agreement.py", "desc": "Oblique bare noun zero agreement", "ref": "NOUN_MARKING_SUFFIXES comment"},
    90:  {"status": "ENCODED", "module": "agreement.py", "desc": "No definiteness agreement in NP", "ref": "docstring, NOUN_MARKING_SUFFIXES"},
    91:  {"status": "N/A",     "module": "",             "desc": "Kurmanci gender agreement (not Sorani)", "ref": ""},
    92:  {"status": "N/A",     "module": "",             "desc": "Kurmanci oblique case (not Sorani)", "ref": ""},
    93:  {"status": "ENCODED", "module": "agreement.py", "desc": "Formal coincidence vs true agreement", "ref": "RECIPROCAL_PRONOUNS comment, _is_invariant docstring"},
    # --- Book 11: Amin (2016) "Verb Grammar of Kurdish" ---
    94:  {"status": "ENCODED", "module": "agreement.py, tense_agreement.py", "desc": "Nominative case unifies both laws", "ref": "docstring"},
    95:  {"status": "ENCODED", "module": "tense_agreement.py", "desc": "Present root 5 classes", "ref": "conjugation constants"},
    96:  {"status": "ENCODED", "module": "agreement.py, subject_verb.py, tense_agreement.py", "desc": "3SG ات allomorph (8 verbs)", "ref": "AT_ALLOMORPH_VERBS"},
    97:  {"status": "THESIS",  "module": "",             "desc": "Optional ت deletion in compound verbs", "ref": "Ch5 §5.3"},
    98:  {"status": "ENCODED", "module": "tense_agreement.py", "desc": "8-tense conjugation formulas", "ref": "PAST/PRESENT_ENDINGS"},
    99:  {"status": "ENCODED", "module": "clitic.py",    "desc": "5 clitic position rules", "ref": "docstring Rules 1-5"},
    100: {"status": "ENCODED", "module": "clitic.py",    "desc": "Double clitic reversal (Germian)", "ref": "docstring F#100"},
    101: {"status": "ENCODED", "module": "tense_agreement.py", "desc": "ویستن exception (double-clitic)", "ref": "EXCEPTIONAL_VERBS"},
    102: {"status": "ENCODED", "module": "tense_agreement.py", "desc": "Passive morpheme decomposition", "ref": "docstring F#102"},
    103: {"status": "THESIS",  "module": "",             "desc": "Conditional 6 laws", "ref": "Ch4 §4.3"},
    104: {"status": "ENCODED", "module": "agreement.py", "desc": "Negation 5 particles", "ref": "VERB_NEGATION_PREFIXES, MOOD_NEG_PREFIX"},
    105: {"status": "THESIS",  "module": "",             "desc": "Perfect continuous aspect", "ref": "Ch4 §4.3"},
    106: {"status": "THESIS",  "module": "",             "desc": "Inchoative aspect", "ref": "Ch4 §4.3"},
    107: {"status": "ENCODED", "module": "agreement.py, clitic.py, tense_agreement.py", "desc": "Passive clitic reassignment (Set 1→Set 2)", "ref": "docstring"},
    108: {"status": "ENCODED", "module": "clitic.py",    "desc": "Directional postbound clitic", "ref": "docstring F#108"},
    109: {"status": "ENCODED", "module": "tense_agreement.py", "desc": "Portmanteau ەتی (3sg+2sg)", "ref": "docstring F#109"},
    110: {"status": "ENCODED", "module": "tense_agreement.py", "desc": "Verb morpheme ordering (14 templates)", "ref": "docstring"},
    111: {"status": "THESIS",  "module": "",             "desc": "Imperative بـ optionality", "ref": "Ch5 §5.3"},
    112: {"status": "ENCODED", "module": "agreement.py", "desc": "Causative ێن/اند", "ref": "CAUSATIVE_SUFFIX_EWE/ANDN"},
    113: {"status": "ENCODED", "module": "agreement.py, tense_agreement.py", "desc": "Modal force subjunctive", "ref": "docstring"},
    114: {"status": "ENCODED", "module": "clitic.py",    "desc": "Dative postbound restrictions", "ref": "docstring F#114"},
    # --- Book 12: Farhadi (2013) ---
    115: {"status": "ENCODED", "module": "agreement.py", "desc": "Definiteness marker migration", "ref": "DEFINITE_MARKER_MIGRATION_DESCRIPTIVE"},
    116: {"status": "ENCODED", "module": "clitic.py",    "desc": "Negative progressive clitic shift", "ref": "docstring F#116"},
    117: {"status": "ENCODED", "module": "agreement.py", "desc": "Pre-head determiners (no ezafe)", "ref": "PRE_HEAD_DETERMINERS"},
    118: {"status": "ENCODED", "module": "agreement.py", "desc": "Complement vs adjunct verbs", "ref": "COMPLEMENT_REQUIRING_VERBS"},
    119: {"status": "ENCODED", "module": "agreement.py", "desc": "Interrogative sentence formation", "ref": "QUESTION_WORDS, YESNO_QUESTION_PARTICLES"},
    120: {"status": "ENCODED", "module": "clitic.py",    "desc": "Passive transitivity prerequisite", "ref": "docstring F#120"},
    121: {"status": "ENCODED", "module": "agreement.py", "desc": "Negative concord (هیچ + neg verb)", "ref": "NEGATIVE_CONCORD_TRIGGERS"},
    122: {"status": "ENCODED", "module": "agreement.py", "desc": "ش/یش ordering asymmetry", "ref": "YISH_BEFORE_CLITIC_ONLY"},
    123: {"status": "ENCODED", "module": "agreement.py", "desc": "Demonstrative+preposition contraction", "ref": "DEMONSTRATIVE_PREPOSITION_CONTRACTIONS"},
    124: {"status": "ENCODED", "module": "agreement.py, clitic.py", "desc": "هی/ئی blocks clitics", "ref": "CLITIC_BARRED_PRONOUNS"},
    125: {"status": "ENCODED", "module": "agreement.py, clitic.py", "desc": "Imperative clitic restrictions", "ref": "IMPERATIVE_MOOD_PREFIXES"},
    126: {"status": "ENCODED", "module": "agreement.py, clitic.py", "desc": "Compound verb clitic insertion", "ref": "COMPOUND_VERB_NOMINAL_ELEMENTS"},
    127: {"status": "ENCODED", "module": "agreement.py", "desc": "Causative suffix clitic position", "ref": "CAUSATIVE_SUFFIX_EWE/ANDN"},
    128: {"status": "ENCODED", "module": "agreement.py", "desc": "Reciprocal plural requirement", "ref": "RECIPROCAL_VARIANTS"},
    129: {"status": "ENCODED", "module": "agreement.py, clitic.py", "desc": "Clitic first-element attachment", "ref": "CLITIC_HOST_CATEGORIES"},
    130: {"status": "ENCODED", "module": "agreement.py, clitic.py", "desc": "Morpheme collision tolerance", "ref": "PERMITTED_MORPHEME_DOUBLINGS"},
    131: {"status": "ENCODED", "module": "agreement.py", "desc": "Derivational-before-grammatical ordering", "ref": "DERIV_BEFORE_GRAM_RULE_INVIOLABLE"},
    132: {"status": "ENCODED", "module": "agreement.py", "desc": "Preverb transitivity flip", "ref": "PREVERB_TRANSITIVITY_FLIPS"},
    133: {"status": "ENCODED", "module": "agreement.py", "desc": "Same-set clitic exclusion", "ref": "SAME_SET_CLITIC_EXCLUSION_IN_SIMPLE"},
    134: {"status": "ENCODED", "module": "agreement.py, analyzer.py", "desc": "Definiteness ەکە verb attachment ban", "ref": "DEFINITENESS_VERB_ATTACHMENT_BAN"},
    135: {"status": "ENCODED", "module": "agreement.py", "desc": "Morpheme ە 5 functions", "ref": "MORPHEME_E_FUNCTIONS"},
    136: {"status": "ENCODED", "module": "agreement.py", "desc": "Definiteness precedes all gram suffixes", "ref": "DEFINITENESS_PRECEDES_ALL_GRAM_SUFFIXES"},
    137: {"status": "ENCODED", "module": "agreement.py", "desc": "Preposition-case role mapping", "ref": "PREPOSITION_CASE_MAP"},
    138: {"status": "ENCODED", "module": "agreement.py", "desc": "Object selection constraints", "ref": "OBJECTIVISABLE_ROLES"},
    139: {"status": "ENCODED", "module": "agreement.py", "desc": "7-role subject selection hierarchy", "ref": "SUBJECT_SELECTION_HIERARCHY"},
    140: {"status": "ENCODED", "module": "agreement.py", "desc": "Passive subject promotion restriction", "ref": "PASSIVE_SUBJECTIVISABLE_ROLES"},
    141: {"status": "ENCODED", "module": "agreement.py", "desc": "Relative clause کە deletion rules", "ref": "RELATIVE_CLAUSE_MARKER"},
    142: {"status": "THESIS",  "module": "",             "desc": "Preposition removal cascading", "ref": "Ch5 §5.3"},
    # --- F#143–F#186: Documented blocks (Mukriani, Farhadi, Rasul, etc.) ---
    143: {"status": "DOCBLOCK", "module": "agreement.py", "desc": "Definiteness in coordinated vs ezafe NPs", "ref": "COORDINATION_CONJUNCTION"},
    144: {"status": "DOCBLOCK", "module": "agreement.py", "desc": "Quantifier position controls verb", "ref": "ORDINAL_SUFFIXES"},
    145: {"status": "DOCBLOCK", "module": "agreement.py", "desc": "Direct object precedes verb", "ref": "DIRECT_OBJECT_PRECEDES_VERB"},
    146: {"status": "DOCBLOCK", "module": "agreement.py", "desc": "Intransitive complement بە", "ref": "INTRANSITIVE_COMPLEMENT_PREP"},
    147: {"status": "DOCBLOCK", "module": "agreement.py", "desc": "Vocative marker on common noun only", "ref": "VOCATIVE_SUFFIXES"},
    148: {"status": "THESIS",  "module": "",             "desc": "10 verb-deletion contexts", "ref": "Ch4 §4.4"},
    149: {"status": "DOCBLOCK", "module": "agreement.py", "desc": "Conditional بـ sole intervenor", "ref": "CONDITIONAL_CLITIC_INTERVENOR"},
    150: {"status": "DOCBLOCK", "module": "agreement.py", "desc": "V-S inversion restricted to copular", "ref": "SUBJECT_PRECEDES_VERB"},
    151: {"status": "DOCBLOCK", "module": "agreement.py", "desc": "Conjoined NP modifier scope ambiguity", "ref": "CONJOINED_NP_MODIFIER_AMBIGUITY"},
    152: {"status": "DOCBLOCK", "module": "agreement.py", "desc": "IO = adverbial, non-promotable", "ref": "IO_PROMOTES_TO_PASSIVE_SUBJECT"},
    153: {"status": "DOCBLOCK", "module": "agreement.py", "desc": "Adverbial positional freedom", "ref": "ADVERBIAL_POSITION_FREE"},
    154: {"status": "DOCBLOCK", "module": "agreement.py", "desc": "Commissive verbs 1st person only", "ref": "COMMISSIVE_VERBS"},
    155: {"status": "DOCBLOCK", "module": "agreement.py", "desc": "Dialectal participle morpheme variation", "ref": "PARTICIPLE_MORPHEME_STANDARD"},
    156: {"status": "THESIS",  "module": "",             "desc": "Adversative paired connectors", "ref": "Ch4 §4.4"},
    157: {"status": "DOCBLOCK", "module": "agreement.py", "desc": "Three-mood tense-morpheme mapping", "ref": "MOOD_TENSE_MAP"},
    158: {"status": "DOCBLOCK", "module": "agreement.py", "desc": "Optative paradigm template", "ref": "OPTATIVE_EXTENDED_SUFFIX"},
    159: {"status": "DOCBLOCK", "module": "agreement.py", "desc": "Patient vs agent participle tense restriction", "ref": "PATIENT_PARTICIPLE_MORPHEME"},
    160: {"status": "DOCBLOCK", "module": "agreement.py", "desc": "Verb class dual-membership", "ref": "DUAL_CLASS_INTRANSITIVE_VERBS"},
    161: {"status": "DOCBLOCK", "module": "agreement.py", "desc": "7 speech act types → punctuation", "ref": "SPEECH_ACT_PUNCTUATION"},
    162: {"status": "DOCBLOCK", "module": "agreement.py", "desc": "Polite imperative preparatory phrases", "ref": "POLITE_IMPERATIVE_MARKERS"},
    163: {"status": "DOCBLOCK", "module": "agreement.py", "desc": "Allophonic pairs ح/ع↔هـ, خ↔غ", "ref": "ALLOPHONIC_PAIRS"},
    164: {"status": "DOCBLOCK", "module": "agreement.py", "desc": "و/وو short-vs-long confusion", "ref": "SHORT_LONG_VOWEL_PAIRS"},
    165: {"status": "ENCODED", "module": "agreement.py, analyzer.py, agreement_accuracy.py", "desc": "ی/یی six scenarios", "ref": "YI_DOUBLE_SCENARIOS"},
    166: {"status": "DOCBLOCK", "module": "agreement.py", "desc": "POS deletability hierarchy", "ref": "CONSTITUENT_DELETABILITY_HIERARCHY"},
    167: {"status": "DOCBLOCK", "module": "agreement.py", "desc": "Epenthetic ت insertion", "ref": "EPENTHETIC_T_ENVIRONMENTS"},
    168: {"status": "DOCBLOCK", "module": "agreement.py", "desc": "خواردن exception to causative ا→ێ", "ref": "CAUSATIVE_A_TO_E_EXCEPTIONS"},
    169: {"status": "DOCBLOCK", "module": "agreement.py", "desc": "ە+ە→ا phonological assimilation", "ref": "VOWEL_COALESCENCE_RULES"},
    170: {"status": "DOCBLOCK", "module": "agreement.py", "desc": "Compound verb morpheme slot patterns", "ref": "COMPOUND_VERB_PREVERBAL_ELEMENTS"},
    171: {"status": "DOCBLOCK", "module": "agreement.py", "desc": "Default constituent ordering SOV", "ref": "DEFAULT_CONSTITUENT_ORDER"},
    172: {"status": "DOCBLOCK", "module": "agreement.py", "desc": "Attributive ی blocks determiners with proper nouns", "ref": "ATTRIBUTIVE_EZAFE_PROPER_NOUN_RULE"},
    173: {"status": "DOCBLOCK", "module": "agreement.py", "desc": "Superlative ترین pre-nominal position", "ref": "SUPERLATIVE_POSITION"},
    174: {"status": "DOCBLOCK", "module": "agreement.py", "desc": "ەوە postposition for source verbs", "ref": "SOURCE_DIRECTION_VERBS"},
    175: {"status": "THESIS",  "module": "",             "desc": "Coordination semantic compatibility", "ref": "Ch4 §4.4"},
    176: {"status": "DOCBLOCK", "module": "agreement.py", "desc": "Title nouns (proper-noun specifiers)", "ref": "TITLE_NOUNS"},
    177: {"status": "THESIS",  "module": "",             "desc": "No morphological case in Sorani", "ref": "Ch4 §4.2"},
    178: {"status": "DOCBLOCK", "module": "agreement.py", "desc": "Possessive ی obligatory", "ref": "POSSESSIVE_EZAFE_OBLIGATORY"},
    179: {"status": "DOCBLOCK", "module": "agreement.py", "desc": "Possessor must be [+definite]", "ref": "POSSESSOR_REQUIRES_DEFINITENESS"},
    180: {"status": "DOCBLOCK", "module": "agreement.py", "desc": "Number and quantifier mutually exclusive", "ref": "QUANTIFIERS"},
    181: {"status": "DOCBLOCK", "module": "agreement.py", "desc": "PP inseparable", "ref": "PP_INSEPARABLE"},
    182: {"status": "DOCBLOCK", "module": "agreement.py", "desc": "Attributive adj blocks determiners", "ref": "ATTRIBUTIVE_ADJ_BLOCKS_DETERMINERS"},
    183: {"status": "DOCBLOCK", "module": "agreement.py", "desc": "Bare common noun cannot be NP alone", "ref": "BARE_NOUN_REQUIRES_DETERMINATION"},
    184: {"status": "DOCBLOCK", "module": "agreement.py", "desc": "Abstract nouns resist determiners", "ref": "ABSTRACT_NOUN_RESISTS_DETERMINERS"},
    185: {"status": "DOCBLOCK", "module": "agreement.py", "desc": "Demonstrative blocks proper noun", "ref": "DEMONSTRATIVE_BLOCKS_PROPER_NOUN"},
    186: {"status": "DOCBLOCK", "module": "agreement.py", "desc": "Determiner allomorphs (phonological)", "ref": "DETERMINER_ALLOMORPHS"},
    # --- Later-book findings (F#187–F#256) ---
    187: {"status": "THESIS",  "module": "",             "desc": "Preposition→postposition conversion table", "ref": "Ch4 §4.5"},
    188: {"status": "THESIS",  "module": "",             "desc": "3SG 12 allomorphs", "ref": "Ch4 §4.3"},
    189: {"status": "ENCODED", "module": "clitic.py",    "desc": "7 clitic laws (systematic rules)", "ref": "docstring"},
    190: {"status": "ENCODED", "module": "clitic.py",    "desc": "Triple clitic ordering", "ref": "docstring F#190"},
    191: {"status": "ENCODED", "module": "agreement.py", "desc": "Germian dialect morpheme reversal", "ref": "F#253 block"},
    192: {"status": "THESIS",  "module": "",             "desc": "Unified present root rule", "ref": "Ch4 §4.3"},
    193: {"status": "THESIS",  "module": "",             "desc": "Stress disambiguates compound verb", "ref": "Ch4 §4.3"},
    194: {"status": "ENCODED", "module": "tense_agreement.py", "desc": "False agreement marker ویستن", "ref": "EXCEPTIONAL_VERBS (F#101)"},
    195: {"status": "THESIS",  "module": "",             "desc": "ڕ epenthesis", "ref": "Ch4 §4.3"},
    196: {"status": "ENCODED", "module": "agreement.py", "desc": "ت epenthesis in perfect", "ref": "EPENTHETIC_T_ENVIRONMENTS (F#167)"},
    197: {"status": "ENCODED", "module": "tense_agreement.py", "desc": "تی portmanteau (3sg+2sg)", "ref": "F#109 in docstring"},
    198: {"status": "ENCODED", "module": "agreement.py", "desc": "Possessive هەبوون paradigm", "ref": "EXISTENTIAL_STEMS comment F#72"},
    199: {"status": "THESIS",  "module": "",             "desc": "Passive of intransitive", "ref": "Ch4 §4.3"},
    200: {"status": "ENCODED", "module": "agreement.py", "desc": "Compound verb 13 patterns", "ref": "COMPOUND_VERB_PREVERBAL_ELEMENTS (F#170)"},
    201: {"status": "THESIS",  "module": "",             "desc": "Morphological ambiguity catalogue", "ref": "Ch4 §4.2"},
    202: {"status": "THESIS",  "module": "",             "desc": "Vocative adjective agreement", "ref": "Ch4 §4.5"},
    203: {"status": "THESIS",  "module": "",             "desc": "Gender marker paradigm (Arabic loans)", "ref": "Ch4 §4.2"},
    204: {"status": "ENCODED", "module": "clitic.py",    "desc": "Clitic omission = ungrammatical", "ref": "docstring F#204"},
    205: {"status": "THESIS",  "module": "",             "desc": "Inanimate gender neutralisation", "ref": "Ch4 §4.2"},
    206: {"status": "THESIS",  "module": "",             "desc": "Post-head determiners nominative only", "ref": "Ch4 §4.5"},
    207: {"status": "ENCODED", "module": "agreement.py", "desc": "Weak verb agreement", "ref": "EXISTENTIAL_STEMS comment"},
    208: {"status": "ENCODED", "module": "clitic.py",    "desc": "7+7 clitic role table", "ref": "docstring"},
    209: {"status": "THESIS",  "module": "",             "desc": "Present tense markers", "ref": "Ch4 §4.3"},
    210: {"status": "THESIS",  "module": "",             "desc": "Aspect decomposition", "ref": "Ch4 §4.3"},
    211: {"status": "THESIS",  "module": "",             "desc": "Copula بوون three roots", "ref": "Ch4 §4.3"},
    212: {"status": "THESIS",  "module": "",             "desc": "Onomatopoeia verb derivation", "ref": "Ch4 §4.2"},
    213: {"status": "THESIS",  "module": "",             "desc": "Syntactic causative", "ref": "Ch4 §4.3"},
    214: {"status": "THESIS",  "module": "",             "desc": "Imperative ە epenthesis", "ref": "Ch4 §4.3"},
    215: {"status": "THESIS",  "module": "",             "desc": "Past conditional clitic ordering", "ref": "Ch4 §4.3"},
    216: {"status": "THESIS",  "module": "",             "desc": "Negative imperative مە", "ref": "Ch4 §4.3"},
    # --- Preposition/conjunction findings (Books 27-30) ---
    217: {"status": "ENCODED", "module": "agreement.py", "desc": "Complete preposition inventory", "ref": "SORANI_SIMPLE_PREPOSITIONS"},
    218: {"status": "ENCODED", "module": "agreement.py", "desc": "لە polysemy (7 functions)", "ref": "LE_POLYSEMY"},
    219: {"status": "ENCODED", "module": "agreement.py", "desc": "Circumposition pattern لە+NP+دا", "ref": "LE_POLYSEMY (F#250)"},
    220: {"status": "THESIS",  "module": "",             "desc": "لەگەڵ polysemy", "ref": "Ch4 §4.5"},
    221: {"status": "ENCODED", "module": "agreement.py", "desc": "Preposition substitutable pairs", "ref": "PREPOSITION_SUBSTITUTABLE_PAIRS"},
    222: {"status": "ENCODED", "module": "agreement.py", "desc": "بۆ polysemy (4 functions)", "ref": "BO_POLYSEMY"},
    223: {"status": "ENCODED", "module": "agreement.py", "desc": "Privative/limitative prepositions", "ref": "PRIVATIVE_PREPOSITIONS, LIMITATIVE_PREPOSITIONS"},
    224: {"status": "ENCODED", "module": "agreement.py", "desc": "بە postbound patterns", "ref": "BA_POSTBOUND"},
    225: {"status": "ENCODED", "module": "agreement.py", "desc": "Nominal prep spatial/temporal duality", "ref": "NOMINAL_PREP_SPATIAL_FRAME"},
    226: {"status": "ENCODED", "module": "agreement.py", "desc": "Free-standing vocative particles", "ref": "VOCATIVE_PARTICLES"},
    227: {"status": "ENCODED", "module": "agreement.py", "desc": "Emphatic superlative هەرە+تر+ترین", "ref": "EMPHATIC_SUPERLATIVE_PREFIX"},
    228: {"status": "ENCODED", "module": "agreement.py", "desc": "Purpose subordinators", "ref": "PURPOSE_SUBORDINATORS"},
    229: {"status": "ENCODED", "module": "agreement.py", "desc": "بەڵکوو correlative requirements", "ref": "BALKUU_CORRELATIVES"},
    230: {"status": "ENCODED", "module": "agreement.py", "desc": "Complete conjunction inventory", "ref": "SORANI_COORDINATING_CONJUNCTIONS, SORANI_SUBORDINATING_CONJUNCTIONS"},
    231: {"status": "ENCODED", "module": "agreement.py", "desc": "و/وو-final noun definite allomorph", "ref": "DEFINITENESS_ALLOMORPHS"},
    232: {"status": "ENCODED", "module": "agreement.py", "desc": "Coordinated NP plural on last only", "ref": "COORDINATED_NP_PLURAL_LAST_ONLY"},
    233: {"status": "ENCODED", "module": "agreement.py", "desc": "ش conjunction requires distinct subjects", "ref": "SH_CONJUNCTION_REQUIRES_DISTINCT_SUBJECTS"},
    234: {"status": "ENCODED", "module": "agreement.py", "desc": "Interrogative چ forces indefinite", "ref": "INTERROGATIVE_CH_FORCES_INDEFINITE"},
    235: {"status": "ENCODED", "module": "agreement.py", "desc": "Optative sentence particles", "ref": "OPTATIVE_SENTENCE_PARTICLES"},
    236: {"status": "ENCODED", "module": "agreement.py", "desc": "لەگەڵ first-noun agreement only", "ref": "COMITATIVE_PREPOSITION"},
    237: {"status": "ENCODED", "module": "agreement.py", "desc": "Tense concordance in compound sentences", "ref": "SEQUENTIAL_CONJUNCTIONS"},
    238: {"status": "ENCODED", "module": "agreement.py", "desc": "Paired conjunctions scope rule", "ref": "PAIRED_CONJUNCTIONS"},
    239: {"status": "ENCODED", "module": "agreement.py", "desc": "Obligatory ellipsis in compound sentences", "ref": "COMPOUND_ELLIPSIS_OBLIGATORY"},
    240: {"status": "ENCODED", "module": "agreement.py", "desc": "بۆ interrogative 'why' context", "ref": "BO_INTERROGATIVE_CONTEXT"},
    241: {"status": "ENCODED", "module": "agreement.py", "desc": "Dual-function connectives", "ref": "DUAL_FUNCTION_CONNECTIVES"},
    242: {"status": "ENCODED", "module": "agreement.py", "desc": "Preposition postfix allomorphy", "ref": "PREPOSITION_POSTFIX_ALLOMORPHS"},
    243: {"status": "ENCODED", "module": "agreement.py", "desc": "ش/یش 4 attachment positions", "ref": "SHISH_POSITION_COUNT"},
    244: {"status": "ENCODED", "module": "agreement.py", "desc": "نە within و-coordination → plural", "ref": "NEGATION_WITHIN_WA_COORDINATION_PLURAL"},
    245: {"status": "ENCODED", "module": "agreement.py", "desc": "Preposition-to-bound-form table", "ref": "PREPOSITION_BOUND_FORMS"},
    246: {"status": "ENCODED", "module": "agreement.py", "desc": "ش/یش conjunction vs marker diagnostic", "ref": "SHISH_REMOVABLE_IS_MARKER"},
    247: {"status": "ENCODED", "module": "agreement.py", "desc": "Postfix scoping last conjunct only", "ref": "PREPOSITION_POSTFIX_LAST_CONJUNCT_ONLY"},
    248: {"status": "ENCODED", "module": "agreement.py", "desc": "Demonstrative circumfix blocks definite article", "ref": "DEMONSTRATIVE_BLOCKS_DEFINITE_ARTICLE"},
    249: {"status": "THESIS",  "module": "",             "desc": "Duration preposition بۆ optional", "ref": "DURATION_PREPOSITION_OPTIONAL"},
    250: {"status": "ENCODED", "module": "agreement.py", "desc": "Circumposition pattern (postposition obligatory)", "ref": "LE_POLYSEMY (F#218)"},
    251: {"status": "THESIS",  "module": "",             "desc": "Preposition bound form diagnostic", "ref": "Ch4 §4.5"},
    252: {"status": "ENCODED", "module": "agreement.py", "desc": "Object pro-drop constraint", "ref": "PRO_DROP_OBJECT_EXCEPTION_VERBS"},
    253: {"status": "ENCODED", "module": "agreement.py", "desc": "Germian dialect morpheme reversal", "ref": "GERMIAN_MORPHEME_ORDER_REVERSED"},
    254: {"status": "ENCODED", "module": "agreement.py, agreement_accuracy.py", "desc": "Asymmetric tense sequencing in و-clauses", "ref": "COORD_TENSE_SEQ_BLOCKED"},
    255: {"status": "ENCODED", "module": "agreement.py", "desc": "مەگەر subjunctive-only; ئەگەر both moods", "ref": "MEGER_MOOD"},
    256: {"status": "ENCODED", "module": "builder.py, constants.py", "desc": "هەرگیز tense restriction", "ref": "HERGIZ_ADVERBS"},
}


