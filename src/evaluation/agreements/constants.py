__all__ = ['logging', 'dataclass', 'field', 'Optional', 'MorphologicalAnalyzer', 'CLITIC_PERSON_MAP', 'SUBJECT_PRONOUNS', 'TRANSITIVE_PAST_STEMS', 'CLITIC_BARRED_PRONOUNS', 'RECIPROCAL_VARIANTS', '_is_present_verb', '_is_transitive_past', '_is_past_verb', 'build_agreement_graph', 'logging', 'logger', '_PRONOUN_AGREEMENT', '_PRESENT_PREFIXES', '_NEGATION_PRESENT_PREFIX', '_NEGATION_PAST_PREFIX', '_IMPERATIVE_PREFIX', '_DEM_FULL_FORMS', '_PROPER_PLACE_NAMES', '_PRESENT_STEMS_NON_B', '_IMPERATIVE_E_STEMS', '_INTRANS_PRESENT_STEMS', '_SH_ZH_PAST_STEM_PREFIXES', '_PAST_INTRANS_STEMS_CHECK', '_PRONOUN_PAST_SUFFIX', '_SUBJ_CONTEXT_OK', '_FAMILIARITY_RANK', '_RUUDAN_PRESENT_STEMS_CHECK', '_SUPPLETIVE_CAUSATIVES', '_XWARDIN_ERRORS', '_PREVERBS', '_PREVERB_NOMINAL_EXCLUSIONS', '_PLURAL_VOCATIVES', '_NUMERALS_PLURAL', '_TIME_NOUNS', '_DEM_FRAME_STOP', '_DEM_FUSED_TIME', '_ORTHO_CONFUSIONS', '_NEG_MARKERS', '_QUANTIFIERS_PLURAL', '_REL_MARKERS', '_VOCATIVE_MARKERS', '_PAST_ADVERBS', '_PRESENT_ADVERBS', '_PERSON_HIERARCHY', '_BARE_NOUN_INDICATORS', '_PRESENT_ENDINGS', '_CHECK_LABELS', '_PAST_INTRANS_STEMS_CHECK', '_PAST_SET2_SUFFIXES', '_PRONOUN_PAST_SUFFIX', '_SUBJ_CONTEXT_OK', '_FAMILIARITY_RANK', '_RUUDAN_PRESENT_STEMS_CHECK', '_SUPPLETIVE_CAUSATIVES', '_XWARDIN_ERRORS', '_PREVERBS', '_PREVERB_NOMINAL_EXCLUSIONS', '_PLURAL_VOCATIVES', '_NUMERALS_PLURAL', '_TIME_NOUNS', '_DEM_FRAME_STOP', '_DEM_FUSED_TIME']
import logging
from dataclasses import dataclass, field
from typing import Optional
from ...morphology.analyzer import (
    MorphologicalAnalyzer,
    CLITIC_PERSON_MAP,
)
from ...morphology.constants import (
    SUBJECT_PRONOUNS,
    TRANSITIVE_PAST_STEMS,
    CLITIC_BARRED_PRONOUNS,
    RECIPROCAL_VARIANTS,
)
from ...morphology.builder import (
    _is_present_verb,
    _is_transitive_past,
    _is_past_verb,
    build_agreement_graph,
)

import logging
logger = logging.getLogger(__name__)

_PRONOUN_AGREEMENT = SUBJECT_PRONOUNS
_PRESENT_PREFIXES = ("دە", "ئە")
_NEGATION_PRESENT_PREFIX = "نا"
_NEGATION_PAST_PREFIX = "نە"
_IMPERATIVE_PREFIX = "ب"
_DEM_FULL_FORMS = ("ئەمانە", "ئەوانە", "ئەمە", "ئەوە", "ئەم", "ئەو")
_PROPER_PLACE_NAMES = (
    "هەولێر", "دهۆک", "کەرکووک", "زاخۆ", "هەڵەبجە", "کۆیە", "ڕانیە",
    "بەغدا", "کوردستان", "عێراق",
)
_PRESENT_STEMS_NON_B = (
    "نووس", "خوێن", "فرۆش", "کوژ", "نێر", "زان", "کڕ", "گر",
    "کەو", "خۆ", "ڕۆ", "دە", "کە", "چ",
)
_IMPERATIVE_E_STEMS = ("نووس", "خوێن", "فرۆش", "کوژ", "نێر", "زان", "کڕ", "گر")
_INTRANS_PRESENT_STEMS = ("کەو", "چ", "ڕۆ", "خەو")
_SH_ZH_PAST_STEM_PREFIXES = ("کوش", "هاوێش")
_PAST_INTRANS_STEMS_CHECK = (
    "ڕۆیشت", "گەیشت", "نووست", "هەستا", "دانیشت", "کەوت", "هات",
    "چوو", "مرد", "فڕی", "گریا", "ترسا", "خەوت", "ژیا",
)
_PRONOUN_PAST_SUFFIX = {
    ("1", "sg"): "م", ("2", "sg"): "یت", ("3", "sg"): "",
    ("1", "pl"): "ین", ("2", "pl"): "ن", ("3", "pl"): "ن",
}
_SUBJ_CONTEXT_OK = {
    "بەڵام", "کە", "چونکە", "ئەگەر", "کاتێک", "پاشان", "ئینجا",
    "دوێنێ", "ئەمڕۆ", "ئێستا", "بۆیە", "،", ".", "؟", "!",
}
_FAMILIARITY_RANK = {
    "من": 1, "ئێمە": 1, "منیش": 1, "ئێمەش": 1,
    "تۆ": 2, "ئێوە": 2, "تۆش": 2, "تۆیش": 2, "ئێوەش": 2,
    "ئەو": 3, "ئەوان": 3, "ئەویش": 3, "ئەوانیش": 3,
}
_RUUDAN_PRESENT_STEMS_CHECK = ("سووت", "شک", "خنک", "پس", "ڕژ")
_SUPPLETIVE_CAUSATIVES = (
    ("هاتاند", "هێنا"), ("چوواند", "برد"), ("ڕۆیشتاند", "نارد"),
    ("کەوتاند", "خست"), ("نووستاند", "نواند"),
)
_XWARDIN_ERRORS = {
    "دەخوێم": "دەخۆم", "دەخوێی": "دەخۆی", "دەخوێین": "دەخۆین",
    "دەخوێت": "دەخوات", "ئەخوێم": "ئەخۆم", "ئەخوێی": "ئەخۆی",
    "ناخوێم": "ناخۆم",
}
_PREVERBS = ("هەڵ", "دا", "ڕا", "دەر", "وەر", "تێ", "لێ", "پێ")
_PREVERB_NOMINAL_EXCLUSIONS = ("تێبینی", "دابین", "هەڵوێست", "پێویست",
                               "ڕاوێژ")
_PLURAL_VOCATIVES = {"کوڕینە", "کچینە", "هاوڕێینە", "خەڵکینە",
                     "براینە", "خوشکینە"}
_NUMERALS_PLURAL = ("دوو", "سێ", "چوار", "پێنج", "شەش", "حەوت",
                    "هەشت", "نۆ", "دە")
_TIME_NOUNS = ("ڕۆژ", "شەو", "ساڵ", "مانگ", "هەفتە", "کاتژمێر",
               "خولەک", "چرکە", "جار", "ساعات", "دەقیقە")
_DEM_FRAME_STOP = {
    "من", "تۆ", "ئەو", "ئێمە", "ئێوە", "ئەوان", "خۆی", "خۆم", "خۆت",
    "یەک", "هەموو", "هەندێک", "چەند", "زۆر", "کەم", "هەر", "هیچ",
    "دوو", "سێ", "چوار", "پێنج", "شەش", "حەوت", "هەشت", "نۆ", "دە",
    "دکتۆر", "مامۆستا", "پرۆفیسۆر", "شێخ", "مەلا", "حاجی", "کاک",
    "خاتوو", "و", "کە", "یان", "بەڵام",
}
_DEM_FUSED_TIME = {"ساڵ": "ئەمساڵ", "شەو": "ئەمشەو", "ڕۆ": "ئەمڕۆ",
                   "جار": "ئەمجارە"}


# Class-level constants extracted

# Class-level constants extracted
# Common orthographic confusion pairs (subset of what orthography.py generates)
_ORTHO_CONFUSIONS = [
    ("ح", "ه"),
    ("خ", "غ"),
    ("ڵ", "ل"),
    ("ڕ", "ر"),
    ("ع", "ئ"),
]

_NEG_MARKERS = {"نە", "نا", "هیچ", "هەرگیز"}

# Sorani quantifiers that force plural agreement on the verb
# (Slevanayi 2001, pp. 87-88; Maaruf 2010, p. 139)
_QUANTIFIERS_PLURAL = {"هەر", "هیچ", "هەموو", "چەند", "هەندێک"}

# Sorani relative clause markers
_REL_MARKERS = {"کە", "ئەوەی"}

# Vocative markers and imperative detection
_VOCATIVE_MARKERS = {"ئەی", "یا"}

# Temporal adverbs with tense constraints
_PAST_ADVERBS = {"دوێنێ", "پار", "پێشتر", "بەرلە", "پارێ"}

_PRESENT_ADVERBS = {"ئێستا", "ئەمڕۆ", "دواتر", "هەمیشە"}

# Person hierarchy for compound subjects: 1st > 2nd > 3rd
_PERSON_HIERARCHY = {"1": 3, "2": 2, "3": 1}

# Common bare nouns (non-pronominal) — treated as 3sg for agreement
_BARE_NOUN_INDICATORS = {"پیاو", "ژن", "منداڵ", "مامۆستا", "قوتابی", "کچ", "کوڕ"}

# Present-tense verb ending -> (person, number)
_PRESENT_ENDINGS: dict[str, tuple[str, str]] = {
    'م':   ('1', 'sg'),   # Set 2: 1sg
    'ەم':  ('1', 'sg'),   # Set 2: 1sg (epenthetic)
    'یت':  ('2', 'sg'),   # Set 2: 2sg
    'ێت':  ('3', 'sg'),   # Set 2: 3sg
    'ات':  ('3', 'sg'),   # Set 2: 3sg (after -a stems)
    'ێ':   ('3', 'sg'),   # Set 2: 3sg (short form)
    'ین':  ('1', 'pl'),   # Set 2: 1pl
    'ن':   ('3', 'pl'),   # Set 2: 3pl
    'ەن':  ('3', 'pl'),   # Set 2: 3pl (epenthetic)
}


_CHECK_LABELS: list[tuple[str, str]] = [
    ("subject_verb", "Law 1"),     # Check 1
    ("clitic_consistency", ""),     # Check 2
    ("ezafe", ""),                  # Check 3
    ("tense_consistency", ""),      # Check 4
    ("object_verb_ergative", "Law 2"),  # Check 5
    ("negative_concord", ""),       # Check 6
    ("orthography", ""),            # Check 7
    ("conditional", ""),            # Check 8
    ("quantifier_noun", ""),        # Check 9
    ("relative_clause", ""),        # Check 10
    ("vocative_imperative", ""),    # Check 11
    ("adverb_verb_tense", ""),      # Check 12
    ("compound_subject", ""),       # Check 13
    ("bare_noun", ""),              # Check 14
    ("noun_subject_verb_number", "Law 1"),  # Check 15
]


_PAST_INTRANS_STEMS_CHECK = (
    "ڕۆیشت", "گەیشت", "نووست", "هەستا", "دانیشت", "کەوت", "هات",
    "چوو", "مرد", "فڕی", "گریا", "ترسا", "خەوت", "ژیا",
)
_PAST_SET2_SUFFIXES: dict[str, tuple[tuple[str, str], ...]] = {
    "": (("3", "sg"),),
    "م": (("1", "sg"),),
    "ی": (("2", "sg"),),
    "یت": (("2", "sg"),),
    "ین": (("1", "pl"),),
    "ن": (("2", "pl"), ("3", "pl")),
}
_PRONOUN_PAST_SUFFIX = {
    ("1", "sg"): "م", ("2", "sg"): "یت", ("3", "sg"): "",
    ("1", "pl"): "ین", ("2", "pl"): "ن", ("3", "pl"): "ن",
}
_SUBJ_CONTEXT_OK = {
    "بەڵام", "کە", "چونکە", "ئەگەر", "کاتێک", "پاشان", "ئینجا",
    "دوێنێ", "ئەمڕۆ", "ئێستا", "بۆیە", "،", ".", "؟", "!",
}
_FAMILIARITY_RANK = {
    "من": 1, "ئێمە": 1, "منیش": 1, "ئێمەش": 1,
    "تۆ": 2, "ئێوە": 2, "تۆش": 2, "تۆیش": 2, "ئێوەش": 2,
    "ئەو": 3, "ئەوان": 3, "ئەویش": 3, "ئەوانیش": 3,
}
_RUUDAN_PRESENT_STEMS_CHECK = ("سووت", "شک", "خنک", "پس", "ڕژ")
_SUPPLETIVE_CAUSATIVES = (
    ("هاتاند", "هێنا"), ("چوواند", "برد"), ("ڕۆیشتاند", "نارد"),
    ("کەوتاند", "خست"), ("نووستاند", "نواند"),
)
_XWARDIN_ERRORS = {
    "دەخوێم": "دەخۆم", "دەخوێی": "دەخۆی", "دەخوێین": "دەخۆین",
    "دەخوێت": "دەخوات", "ئەخوێم": "ئەخۆم", "ئەخوێی": "ئەخۆی",
    "ناخوێم": "ناخۆم",
}
_PREVERBS = ("هەڵ", "دا", "ڕا", "دەر", "وەر", "تێ", "لێ", "پێ")
_PREVERB_NOMINAL_EXCLUSIONS = ("تێبینی", "دابین", "هەڵوێست", "پێویست",
                               "ڕاوێژ")
_PLURAL_VOCATIVES = {"کوڕینە", "کچینە", "هاوڕێینە", "خەڵکینە",
                     "براینە", "خوشکینە"}
_NUMERALS_PLURAL = ("دوو", "سێ", "چوار", "پێنج", "شەش", "حەوت",
                    "هەشت", "نۆ", "دە")
_TIME_NOUNS = ("ڕۆژ", "شەو", "ساڵ", "مانگ", "هەفتە", "کاتژمێر",
               "خولەک", "چرکە", "جار", "ساعات", "دەقیقە")
_DEM_FRAME_STOP = {
    "من", "تۆ", "ئەو", "ئێمە", "ئێوە", "ئەوان", "خۆی", "خۆم", "خۆت",
    "یەک", "هەموو", "هەندێک", "چەند", "زۆر", "کەم", "هەر", "هیچ",
    "دوو", "سێ", "چوار", "پێنج", "شەش", "حەوت", "هەشت", "نۆ", "دە",
    "دکتۆر", "مامۆستا", "پرۆفیسۆر", "شێخ", "مەلا", "حاجی", "کاک",
    "خاتوو", "و", "کە", "یان", "بەڵام",
}
_DEM_FUSED_TIME = {"ساڵ": "ئەمساڵ", "شەو": "ئەمشەو", "ڕۆ": "ئەمڕۆ",
                   "جار": "ئەمجارە"}
