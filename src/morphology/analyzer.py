"""
Rule-Based Morphological Analyzer for Sorani Kurdish

Extracts structured morphological features from Sorani Kurdish text using
rule-based decomposition informed by Kurdish linguistics literature.

The morpheme slot model and feature inventory for verbs are based on
Amin (2016) "Verb Grammar of the Kurdish Language", p. 62, which
decomposes a Kurdish verb into the following ordered slots:

    [negation] [compound] [mood/aspect] STEM [voice] [person/number] [clitic]

Where:
    negation:  نە- (past) / نا- (present) / مە- (imperative)
    compound:  وەر- / هەڵ- / لێ- / تێ- / دەر- / پێ-
    mood/asp:  دە- (habitual/continuous) / ب- (subjunctive) / ئە- (future)
    voice:     -را (passive) / -اند (causative)
    person:    -م / -یت / -ێت / -ین / -ن  (present set)
               -م / -ت / ∅ / -مان / -تان / -یان  (past/clitic set)

Fatah & Qadir (2006) "Some Aspects of Kurdish Morphology" provide the
theoretical foundation for morpheme decomposition, distinguishing three
levels of analysis (morpheme → morph → phoneme) and classifying Kurdish
as having both agglutinating and fusional morphological properties. Key
results from that work inform this analyzer:
  - Ezafe (ی) is a single-morph single-morpheme unit.
  - Definiteness (-ەکە) and indefiniteness (-ێک) are single-morpheme
    units that can be segmented without semantic loss.
  - Morpheme boundaries are not always phonologically transparent; a
    rule-based segmenter must account for fusional contexts.

Morphophonological findings informing this analyzer:
  F#27  — Double ی in past tense of ی-root verbs (YI_DOUBLE_SCENARIOS)
  F#97  — Optional 2sg/3sg ت deletion in compound verbs
  F#105 — Perfect continuous: لە+Infinitive+دا (always Set 2)
  F#106 — Inchoative aspect: کەوتنە + Infinitive
  F#193 — Stress disambiguates compound-verb tense vs copula + adjective
  F#194 — False agreement in ویستن-derived verbs (3sg -ێت is root, not agr)
  F#195 — ڕ epenthesis between two vowels in compound imperatives
  F#196 — ت epenthesis in present perfect + directional ـە adjacency

Haji Marif (2014) "Kurdish Grammar: Vol. 1, Part 2 (Pronouns)" contributes
the clitic person/number identification, distinguishing first-set (agent)
and second-set (patient) clitics in ergative past-tense constructions.

Closed-class POS detection uses lexicons from agreement.py, sourced from:
  - Slevanayi (2001): pronouns, demonstratives, quantifiers, invariant forms
  - Abbas & Sabir (2020): preposition inventory (Finding #217)
  - Ibrahim (1988): conjunction inventory (Finding #230), optative particles
  - Haji Marf (2014): reciprocal pronouns, question words
"""

import logging
import re
from typing import Optional
from dataclasses import dataclass, field

from .constants import (
    ADJECTIVE_DIMINUTIVE_SUFFIXES,
    ADVERB_DERIVATION_PATTERNS,
    AGENT_NOUN_SUFFIX_CLASSES,
    CAUSATIVE_A_TO_E_MAP,
    CAUSATIVE_BANNED_VERBS,
    CAUSATIVE_SUFFIX_EWE,
    COMPARATIVE_ASSIMILATION_RULES,
    COMPOUND_ORDINAL_LAST_ONLY,
    COMPOUND_VERB_PREVERBAL_ELEMENTS,
    DEMONSTRATIVE_PREPOSITION_CONTRACTIONS,
    DEMONSTRATIVES,
    DIMINUTIVE_NOUN_SUFFIXES,
    EPENTHETIC_T_ENVIRONMENTS,
    EPENTHETIC_T_VERB_STEMS,
    FRACTIONAL_NUMBER_PATTERNS,
    HABUN_EXISTENCE_NEGATION,
    HABUN_POSSESSION_NEGATION,
    HERE_SUPERLATIVE_POST_NOMINAL,
    INDEFINITE_ARTICLE_FORMS,
    INSTRUMENT_NOUN_PREFIX,
    INTERROGATIVE_PRONOUNS,
    INTRANSITIVE_DOUBLET_PAIRS,
    INVARIANT_ADJECTIVES,
    IZAFE_MARKERS,
    MORPHEME_E_FUNCTIONS,
    NEGATION_SLOT_MORPHEMES,
    NIWE_ABSTRACT_CONCRETE,
    NOUN_DERIVATION_SUFFIXES,
    NOUN_DERIVING_PREFIXES,
    NOUN_SINGULAR_AFTER_CARDINAL,
    OBLIGATORY_PREVERB_ROOTS,
    OPTATIVE_SENTENCE_PARTICLES,
    ORDINAL_SUFFIX_LONG,
    ORDINAL_SUFFIX_SHORT,
    ORDINAL_SUFFIXES,
    PASSIVE_VOWEL_CHANGES,
    PAST_STEM_FINAL_SOUNDS,
    PLURAL_AN_PHONOLOGICAL_RULES,
    PP_INSEPARABLE,
    PRESENT_STEM_CONSONANT_MUTATIONS,
    QUANTIFIER_FORMS,
    QUESTION_WORDS,
    RECIPROCAL_PRONOUNS,
    SUBJECT_PRONOUNS,
    SUFFIX_YI_FUNCTIONS,
    SUPERLATIVE_POSITION,
    YESNO_QUESTION_PARTICLES,
    YISH_BEFORE_CLITIC_ONLY,
)

logger = logging.getLogger(__name__)

# Import shared tokenizer — the canonical regex and logic now live in
# src/data/tokenize.py so that every module uses the same tokenization.
from ..data.tokenize import sorani_tokenize as _sorani_tokenize  # noqa: E402

# Prepositions — Abbas & Sabir (2020), pp. 23-26 (Finding #217)
SORANI_SIMPLE_PREPOSITIONS = frozenset({
    "لە", "بە", "بۆ", "بێ", "تا", "هەتا", "تاکو", "تاوەکو", "هەتاوەکو",
    "لەگەڵ", "لەتەک",
})
SORANI_NOMINAL_PREPOSITIONS = frozenset({
    "پێش", "پاش", "تەنیشت", "بەرامبەر",
    "نێوان", "سەرەوە", "ژێرەوە", "لای",
})
SORANI_COMPOUND_PREPOSITIONS = frozenset({
    "بەبێ", "لەبۆ", "لە لایەن", "بەدرێژایی", "بەرەو",
    "لەبەر", "لەپێش", "لەپاش", "لەژێر", "لەسەر",
    "لەتەنیشت", "لەبەرامبەر", "لەنێوان",
})

# Conjunctions — Ibrahim (1988), pp. 75-142 (Finding #230)
SORANI_COORDINATING_CONJUNCTIONS = frozenset({
    "و", "بەڵام", "یا", "ئەوسا", "ئەوجا",
    "بەڵکوو", "وەکوو", "ئینجا", "ئەگینا", "وە",
})
SORANI_SUBORDINATING_CONJUNCTIONS = frozenset({
    "کە", "بۆئەوەی", "کەچی", "هەتاکوو",
    "چونکە", "لەبەرئەوەی کە", "ئەگەر", "گەر", "مەگەر",
    "ئەگەرچی", "بۆیە",
})

# Vocative particle forms — analyzer-specific subset for POS detection
VOCATIVE_PARTICLE_FORMS = {"ئەی", "هۆ", "هێ", "یا", "ۆ"}

# ---------------------------------------------------------------------------
# Punctuation pattern for PUNCT POS detection
# ---------------------------------------------------------------------------
PUNCT_PATTERN = re.compile(r'^[\u0021-\u002F\u003A-\u0040\u005B-\u0060'
                           r'\u007B-\u007E\u060C\u061B\u061F\u0640'
                           r'\u066A-\u066D\u06D4\u2000-\u206F'
                           r'\u2E00-\u2E7F\u3000-\u303F]+$')

# ---------------------------------------------------------------------------
# Numeral detection: Kurdish digits (٠-٩), Arabic-Indic (۰-۹), Latin (0-9)
# and spelled-out Kurdish numerals (expanded beyond QUANTIFIER_FORMS)
# ---------------------------------------------------------------------------
KURDISH_NUMERAL_WORDS = {
    "یەک", "دوو", "سێ", "چوار", "پێنج", "شەش",
    "حەوت", "هەشت", "نۆ", "دە", "یازدە", "دوازدە",
    "سێزدە", "چواردە", "پازدە", "شازدە", "حەڤدە", "هەژدە",
    "نۆزدە", "بیست", "سی", "چل", "پەنجا", "شەست",
    "حەفتا", "هەشتا", "نەوەد", "سەد", "هەزار", "ملیۆن",
}
NUMERAL_DIGIT_PATTERN = re.compile(r'^[0-9٠-٩۰-۹]+$')

# ---------------------------------------------------------------------------
# Adverb lexicon — manner, time, place, degree adverbs
# Built from multiple findings: Hergiz adverbs (F#256), question words
# that function as adverbs, and common Sorani adverbs.
# Excludes words primarily used as adjectives (باش, خراپ, etc.)
# ---------------------------------------------------------------------------
SORANI_ADVERBS = {
    # Manner adverbs
    "زوو", "دوا", "دواتر", "بەخێرایی", "بەهێواشی",
    "بەتەنیا", "بەیەکەوە", "قەت",
    # Time adverbs
    "ئێستا", "دوێنێ", "سبەی", "بەیانی", "ئەمڕۆ", "ئەمشەو",
    "هەمیشە", "بەردەوام", "هەرگیز",
    "پێشتر", "دوایی", "دوایین", "بەرێ", "زووتر",
    # Place adverbs
    "لێرە", "ئەوێ", "لەدەرەوە", "لەناوەوە",
    "لەسەرەوە", "لەژێرەوە", "لەپێشەوە", "لەپاشەوە",
    # Degree adverbs
    "تەواو", "تەنیا", "هەر",
    "زۆرتر", "کەمتر", "هەرچەندە", "بەراستی",
    # Negative/emphatic adverbs
    "هیچکات", "بێگومان", "دلنیایی",
}

# Reflexive pronoun stems — خۆ + clitic
REFLEXIVE_STEM = "خۆ"
REFLEXIVE_PRONOUNS = {"خۆم", "خۆت", "خۆی", "خۆمان", "خۆتان", "خۆیان"}

# Combined preposition set for fast lookup
ALL_PREPOSITIONS = (
    SORANI_SIMPLE_PREPOSITIONS
    | SORANI_NOMINAL_PREPOSITIONS
    | SORANI_COMPOUND_PREPOSITIONS
)

# Combined conjunction set
ALL_CONJUNCTIONS = (
    SORANI_COORDINATING_CONJUNCTIONS
    | SORANI_SUBORDINATING_CONJUNCTIONS
)

# Particles: question, optative, vocative, modal
ALL_PARTICLES = (
    YESNO_QUESTION_PARTICLES
    | OPTATIVE_SENTENCE_PARTICLES
    | VOCATIVE_PARTICLE_FORMS
    | {"دەبێت", "ئەبێت", "با"}  # modal / hortative particles
)


# Morpheme slot components based on Amin (2016), p. 62
# These are used for rule-based verb decomposition when KLPT is unavailable.

NEGATION_PREFIXES = {"نە", "نا", "مە"}
# Compound verb preverbal elements — inline set derived from the structured
# COMPOUND_VERB_PREVERBAL_ELEMENTS constant (F#126, Haji Marf 2014) which
# categorizes them as morphosyntactic_preposition vs adverb_prefix.
# F#355: PREVERB_NO_TRANSITIVITY_CHANGE — preverbs never change
# transitivity; only -اندن does.
COMPOUND_PREFIXES = {"وەر", "هەڵ", "لێ", "تێ", "دەر", "پێ"}
MOOD_ASPECT_PREFIXES = {"دە", "ب", "ئە"}
VOICE_SUFFIXES = {"ڕا", "ڕێ", "اند", "ێن"}  # passive-past, passive-present, causative-past, causative-present

# Pre-sorted prefix/suffix lists (longest-first) — avoids re-sorting on every call
_NEGATION_PREFIXES_SORTED = sorted(NEGATION_PREFIXES, key=len, reverse=True)
_COMPOUND_PREFIXES_SORTED = sorted(COMPOUND_PREFIXES, key=len, reverse=True)
_MOOD_ASPECT_PREFIXES_SORTED = sorted(MOOD_ASPECT_PREFIXES, key=len, reverse=True)
_VOICE_SUFFIXES_SORTED = sorted(VOICE_SUFFIXES, key=len, reverse=True)

# Infinitive suffix — Amin (2016), pp. 144-175
# Sorani infinitives = past stem + allomorph + ن
# Five ending patterns: ان, تن, دن, وون, ین (lexically determined by root)
INFINITIVE_SUFFIX = "ن"

PRESENT_PERSON_SUFFIXES = {"م", "یت", "ێت", "ین", "ن", "ەم", "ەن", "ات", "ێ"}
PAST_PERSON_SUFFIXES = {"م", "ت", "مان", "تان", "یان"}  # 3sg = zero

# Pre-sorted person suffix lists (depend on PRESENT/PAST_PERSON_SUFFIXES above)
_PERSON_SUFFIXES_SORTED = sorted(
    PRESENT_PERSON_SUFFIXES | PAST_PERSON_SUFFIXES, key=len, reverse=True
)
_PRESENT_SUFFIXES_SORTED = sorted(PRESENT_PERSON_SUFFIXES, key=len, reverse=True)

# Portmanteau suffix for 3sg transitive present perfect — Amin (2016), p. 56
# Combines copula + 3sg agreement + transitive marking in one morph.
# Example: بردوویەتی "he has taken it"
# Source: Finding #109
# ەتی is the primary form; یتی is a dialectal variant
PORTMANTEAU_3SG_TRANS_PERFECT = "ەتی"
PORTMANTEAU_3SG_TRANS_PERFECT_ALT = "یتی"

# Past tense morpheme allomorphs — Rule R6 (Books 3, 4)
# Source: Wrya Amin (1986), Finding #25; Rasul (2005), pp. 17–18
# Five allomorphs selected by verb root (lexically determined)
PAST_MORPHEME_ALLOMORPHS = {
    "ا": [  # سوتان, وەستان, شکان, کێڵان, ترسان, گەڕان, ڕژان...
        "سووت", "وەست", "شک", "کێڵ", "ترس", "گەڕ", "ڕژ",
        "پس", "خنک", "دڕ", "بڕ", "فەوت", "هەست", "هێن",
        "پێچ", "کوت", "هەڵکش", "سووڕ", "هەڵس",
        "پێو", "ڕووخ", "شەق", "ئاو", "کش",
        # Additional expanded stems
        "گەڕ", "خلیسک", "ڕاگیر", "نیز", "نەچ",
        "تەرخ", "شکاند", "پژم", "بز", "بزر",
    ],
    "ی": [  # فڕین, خزین, تەقین, نووسین, کڕین...
        "فڕ", "خز", "تەق", "نووس", "کڕ", "ناس", "تاش",
        "دۆز", "بەخش", "چن", "بین", "زان", "توان",
        "پەڕ", "پژم", "بز", "کۆک", "بزڕک",
        # Additional
        "گریا", "خوێند", "ژیا", "کەمتر", "گوێگرت",
    ],
    "و": [  # بوون, چوون, دروون
        "بو", "چو", "درو",
        # Additional
        "ڕو", "لاچو",
    ],
    "ت": [  # کوشتن, گرتن, نوستن (sleep), خەوتن, وتن...
        "کوش", "گر", "نوس", "خەو", "گو", "بەس", "بیس",
        "فرۆش", "هاوش", "پش", "شوش", "ڕش",
        "گەیش", "ڕۆیش", "نیش", "دانیش", "پاراس",
        # Additional
        "ڕاکش", "دامەزر", "هەڵگر",
    ],
    "د": [  # کردن, بردن, مردن, خواردن, ناردن...
        "کر", "بر", "مر", "خوار", "نار", "سپار",
        "ژمار", "بژار", "شار", "خوێن",
        # Additional
        "هاوار", "ئاوار",
    ],
}

# Irregular/suppletive past→present stem mapping (Finding #17, Rule R19)
# Source: Rasul (2005), p. 22; Kurdish Academy (2018), pp. 167-181
# F#378: PRESENT_STEM_IRREGULAR_CLASSES — 7 formal derivation classes
# (zero removal, 3-sound, 4-sound, 5-sound+vowel shift, suppletive, etc.).
IRREGULAR_PRESENT_STEMS = {
    # past_stem → present_stem
    "ڕۆیشت": "ڕۆ",      # go
    "برد": "بە",       # carry
    "کرد": "کە",       # do
    "خوارد": "خۆ",     # eat
    "دا": "دە",        # give
    "شوشت": "شۆ",     # wash
    "هاوشت": "هاوێژ",  # throw
    "کوشت": "کوژ",    # kill (ش→ژ alternation, Rule R19)
    "هێنا": "هێن",     # bring
    "گوت": "ڵێ",       # say
    "دیت": "بین",     # see
    "مرد": "مر",       # die
    "گەیشت": "گە",     # arrive
    # اردن verbs: ا→ێ (Rule R7)
    "سپارد": "سپێر",   # deposit
    "ژمارد": "ژمێر",   # count
    "بژارد": "بژێر",   # choose
    "نارد": "نێر",     # send
    "شارد": "شێر",     # hide
    # Additional irregular stems
    "خست": "خ",        # put/drop
    "گرت": "گر",       # take/hold
    "بوو": "ب",        # be/become
    "پاراست": "پار",   # protect
    "بەست": "بەست",    # tie
    "بیست": "بیست",    # hear
    "ویست": "وێ",      # want
    "فرۆشت": "فرۆش",  # sell
    "کەوت": "کەو",    # fall
    "نووست": "نو",     # sleep
    "خەوت": "خەو",    # dream
    "دۆزی": "دۆز",    # find
}

# ش→ژ alternation stems (Rule R19)
# Verbs with ش before ت in infinitive → ژ in present stem
SH_ZH_ALTERNATION = {
    "کوشت": "کوژ",     # kill: کوشتن → دەکوژم
    "بێشت": "بێژ",     # say (dialectal)
    "ڕشت": "ڕێژ",      # pour: ڕشتن → دەڕێژم
    "هاوشت": "هاوێژ",   # throw: هاوشتن → دەهاوێژم
}

# Precomputed stem sets for O(1) lookup in _score_b_prefix_evidence
_ALL_PRESENT_STEMS = frozenset(IRREGULAR_PRESENT_STEMS.values())
_ALL_PAST_STEMS = frozenset(
    s for stems in PAST_MORPHEME_ALLOMORPHS.values() for s in stems
)
_ALL_SH_ZH_STEMS = frozenset(SH_ZH_ALTERNATION.values())

# Nominal morpheme units — Fatah & Qadir (2006), pp. 24-25
# Each is a single-morpheme unit that can be segmented without semantic loss.
EZAFE_MORPHEME = "ی"         # linking particle (single-morph morpheme)
# F#264: BARE_IZAFE_DIALECTS — Sulaimani standard uses ی/ە izafe;
# bare-izafe dialects omit the marker entirely. This analyzer assumes
# standard Sulaimani orthography.
DEFINITE_SUFFIX = "ەکە"       # definiteness marker (the)
INDEFINITE_SUFFIX = "ێک"      # indefiniteness marker (a/an)
PLURAL_SUFFIX = "ان"          # plural marker (F#278: PLURAL_AN_SINGLE_MORPHEME)
# F#303: DOUBLE_PLURAL_STACKING_PERMITTED — باخاتەکان (باخ+ات+ەکان)
# is valid; Arabic-origin double plural + Kurdish definite plural.
# F#304: GEL_PLURAL_ANIMATE_ONLY — -گەل secondary plural is
# restricted to animate nouns only.
# F#305: VERBAL_NOUN_DEFAULT_FEMININE — verbal nouns (infinitives
# used as nouns) default to feminine gender.
DEMONSTRATIVE_SUFFIX = "ە"    # proximal demonstrative

# Nominal suffix patterns (ordered longest-first for greedy match)
# Source: Fatah & Qadir (2006), pp. 24-25; Slevanayi (2001), pp. 47-48
# F#275: IZAFE_SUBSTITUTION_SEMANTIC_SHIFT — substituting izafe form
# (ی vs ە) changes meaning; the suffix order below preserves this distinction.
# F#134: ەکە precedes all other grammatical suffixes
# F#59: vowel-final stems use یان (not ان) for plural
# F#60: vowel-final stems use یەکە (not ەکە) for definite
NOMINAL_SUFFIXES_ORDERED: list[tuple[str, str, str]] = [
    # (suffix, feature_name, feature_value)
    ("ەکان",  "definiteness", "def"),     # definite + plural
    ("یەکان", "definiteness", "def"),     # vowel-final definite + plural
    ("ەکە",   "definiteness", "def"),     # definite singular
    ("یەکە",  "definiteness", "def"),     # vowel-final definite singular
    ("ێکیان", "definiteness", "indef"),   # indefinite + possessive
    ("ییەک",  "definiteness", "indef"),   # vowel-final indefinite
    ("ێک",    "definiteness", "indef"),   # indefinite
    ("یان",   "number",       "pl"),      # vowel-final plural
    ("ان",    "number",       "pl"),      # plural
]

# Ezafe patterns — F#165: six scenarios for ی/یی
# After consonant: ی   After vowel: یی (doubled)
# Ezafe links: noun-adjective, noun-noun (possession), noun-PP
EZAFE_PATTERN = re.compile(r'^(.+?)(یی|ی)$')

# Clitic person/number mapping (with set distinction)
# Source: Amin (2016), pp. 17-18; Finding #9
# Set 1 (agent in present, patient in past): م/ت/ی/مان/تان/یان
# Set 2 (agreement suffixes on verb): م/یت/ێت/ین/ن
CLITIC_PERSON_MAP: dict[str, tuple[str, str]] = {
    "مان": ("1", "pl"),
    "تان": ("2", "pl"),
    "یان": ("3", "pl"),
    "م":   ("1", "sg"),
    "ت":   ("2", "sg"),
    "ی":   ("3", "sg"),
}

# Verb suffix → (person, number) mapping — static, used by analyzer
_SUFFIX_PERSON_NUMBER: dict[str, tuple[str, str]] = {
    "م": ("1", "sg"),
    "ەم": ("1", "sg"),
    "یت": ("2", "sg"),
    "ت": ("2", "sg"),
    "ێت": ("3", "sg"),
    "ێ": ("3", "sg"),
    "ات": ("3", "sg"),
    "ین": ("1", "pl"),
    "ن": ("3", "pl"),
    "ەن": ("3", "pl"),
    "مان": ("1", "pl"),
    "تان": ("2", "pl"),
    "یان": ("3", "pl"),
}

# Cached set difference: adverbial question words = QUESTION_WORDS - INTERROGATIVE_PRONOUNS
_ADVERBIAL_QUESTION_WORDS = QUESTION_WORDS - INTERROGATIVE_PRONOUNS


@dataclass
class MorphFeatures:
    """Structured morphological features for a Sorani Kurdish token.
    
    Feature inventory based on Amin (2016) for verbal features and
    Saliqanai (2001) for agreement-relevant features (person, number,
    case, definiteness).
    """
    token: str
    lemma: str = ""
    pos: str = ""           # Part of speech
    person: str = ""        # 1, 2, 3
    number: str = ""        # sg, pl
    tense: str = ""         # present, past, future, imperative, infinitive
    aspect: str = ""        # habitual, perfective, perfect — Qader (2017)
    case: str = ""          # nom, obl, ez (ezafe)
    definiteness: str = ""  # def, indef
    transitivity: str = ""  # trans, intrans (Finding #80 Split Ergativity)
    clitic_person: str = "" # 1, 2, 3 (Finding #12 Clitic Sets)
    clitic_number: str = "" # sg, pl
    is_clitic: bool = False
    negated: bool = False          # whether negation prefix is present
    compound_prefix: str = ""      # compound verb prefix (وەر, هەڵ, etc.)
    voice: str = ""                # active, passive, causative
    raw_analysis: dict = field(default_factory=dict)

    def to_vector_indices(self, feature_vocab: dict) -> list[int]:
        """Convert features to integer indices for embedding lookup."""
        indices = []
        for feat_name in ["person", "number", "tense", "aspect", "case", "definiteness", "transitivity", "clitic_person", "clitic_number"]:
            feat_val = getattr(self, feat_name, "")
            key = f"{feat_name}:{feat_val}" if feat_val else f"{feat_name}:UNK"
            indices.append(feature_vocab.get(key, 0))
        return indices


class MorphologicalAnalyzer:
    """Rule-based morphological analyzer for Sorani Kurdish.
    
    Uses lexicon-based closed-class detection and rule-based morpheme
    decomposition for open-class words (verbs, nouns).
    
    entries, 2,141 verb stems) for improved stem validation and
    transitivity detection.  Ahmadi (2021), arXiv:2109.06374.
    """
    
    def __init__(self, use_klpt: bool = False, ahmadi_lexicon: "SoraniLexicon | None" = None):
        self._lexicon = ahmadi_lexicon
        self.use_klpt = use_klpt
        self._stem = None
        self._tokenize = None
        
        if use_klpt:
            self._init_klpt()
    
    def _init_klpt(self):
        """Initialize KLPT toolkit (optional enrichment only)."""
        try:
            from klpt.stem import Stem
            from klpt.tokenize import Tokenize
            self._stem = Stem("Sorani", "Arabic")
            self._tokenize = Tokenize("Sorani", "Arabic")
            logger.info("KLPT initialized (optional enrichment)")
        except (ImportError, Exception) as e:
            logger.debug("KLPT not available: %s", e)
            self.use_klpt = False
            self._stem = None
            self._tokenize = None
    
    def analyze_token(self, token: str) -> MorphFeatures:
        """Analyze a single token and return its morphological features.
        
        Analysis pipeline (in order):
        1. Punctuation check
        2. Closed-class lexicon lookup (pronouns, prepositions, conjunctions,
           adverbs, particles, determiners, numerals)
        3. Adjective detection (invariant list + comparative/superlative suffixes)
        4. Verb slot decomposition — Amin (2016), p. 62
        5. Nominal suffix stripping — Fatah & Qadir (2006), pp. 24-25
        6. Clitic detection
        """
        features = MorphFeatures(token=token)
        
        # 1. Punctuation
        if PUNCT_PATTERN.match(token):
            features.pos = "PUNCT"
            features.lemma = token
            return features
        
        # 2-3. Closed-class detection (pronouns, prepositions, etc.)
        if self._classify_closed_class(token, features):
            return features
        
        # 4. Rule-based verb feature extraction
        self._extract_verb_features(token, features)
        
        # 5. If not identified as verb, try nominal analysis
        if features.pos != "VERB":
            self._extract_nominal_features(token, features)
        
        # 6. Clitic detection
        self._detect_clitic(token, features)
        
        # Optional KLPT enrichment (lemma only, never overrides POS)
        if self._stem is not None and not features.lemma:
            try:
                analysis = self._stem.analyze(token)
                if analysis:
                    features.raw_analysis = analysis
                    if not features.lemma:
                        features.lemma = analysis.get("lemma", token)
            except (ValueError, KeyError, AttributeError):
                pass
        
        # Default: if no POS assigned after all analysis, assume NOUN
        # Most open-class unrecognized words in Sorani Kurdish text are nouns
        if not features.pos:
            features.pos = "NOUN"
            if not features.number:
                features.number = "sg"
        
        if not features.lemma:
            features.lemma = token
        
        return features
    
    def _classify_closed_class(self, token: str, features: MorphFeatures) -> bool:
        """Classify token using closed-class lexicons from agreement.py.
        
        Returns True if the token was identified as a closed-class word
        (no further verb/noun analysis needed). Returns False otherwise.
        
        Sources:
        - Slevanayi (2001): pronouns (pp. 37-48, 77-83), demonstratives (pp. 83-86),
          quantifiers (pp. 87-88), invariant adjectives (pp. 38-48)
        - Abbas & Sabir (2020): prepositions (Finding #217)
        - Ibrahim (1988): conjunctions (Finding #230), optative particles
        - Haji Marf (2014): reciprocal pronouns (pp. 296-297)
        """
        # --- Numerals (digits) ---
        if NUMERAL_DIGIT_PATTERN.match(token):
            features.pos = "NUM"
            features.lemma = token
            return True
        
        # --- Subject pronouns: من، تۆ، ئەو، ئێمە، ئێوە، ئەوان ---
        if token in SUBJECT_PRONOUNS:
            features.pos = "PRON"
            features.person, features.number = SUBJECT_PRONOUNS[token]
            features.lemma = token
            return True
        
        # --- Reflexive pronouns: خۆم، خۆت، خۆی، خۆمان، خۆتان، خۆیان ---
        if token in REFLEXIVE_PRONOUNS:
            features.pos = "PRON"
            features.lemma = REFLEXIVE_STEM
            # Extract person/number from the clitic part
            suffix = token[len(REFLEXIVE_STEM):]
            if suffix in CLITIC_PERSON_MAP:
                features.person, features.number = CLITIC_PERSON_MAP[suffix]
            return True
        
        # --- Interrogative pronouns: کێ، چی، چ، کام ---
        if token in INTERROGATIVE_PRONOUNS:
            features.pos = "PRON"
            features.lemma = token
            return True
        
        # --- Reciprocal pronouns: یەکتر، یەکدی، ئێکدی، ئێکتر ---
        if token in RECIPROCAL_PRONOUNS:
            features.pos = "PRON"
            features.lemma = token
            features.number = "pl"
            return True
        
        # --- Demonstratives: ئەو، ئەم، ئەوە، ئەمە ---
        # Tagged as DET (determiner); standalone pro-form use is contextual
        if token in DEMONSTRATIVES:
            features.pos = "DET"
            features.lemma = token
            # ئەوان/ئەمان are plural forms — already caught by SUBJECT_PRONOUNS
            features.number = "sg"
            return True
        
        # --- Prepositions (ADP) ---
        if token in ALL_PREPOSITIONS:
            features.pos = "ADP"
            features.lemma = token
            return True
        
        # --- Coordinating conjunctions ---
        if token in SORANI_COORDINATING_CONJUNCTIONS:
            features.pos = "CCONJ"
            features.lemma = token
            return True
        
        # --- Subordinating conjunctions ---
        if token in SORANI_SUBORDINATING_CONJUNCTIONS:
            features.pos = "SCONJ"
            features.lemma = token
            return True
        
        # --- Particles (question, optative, vocative, modal) ---
        if token in ALL_PARTICLES:
            features.pos = "PART"
            features.lemma = token
            return True
        
        # --- Spelled-out numerals ---
        if token in KURDISH_NUMERAL_WORDS:
            features.pos = "NUM"
            features.lemma = token
            return True
        
        # --- Quantifiers (as DET) ---
        # Checked after numerals since some overlap (دوو, سێ etc.)
        if token in QUANTIFIER_FORMS:
            features.pos = "DET"
            features.lemma = token
            return True
        
        # --- Adverbs ---
        if token in SORANI_ADVERBS:
            features.pos = "ADV"
            features.lemma = token
            return True
        
        # --- Question words that function as adverbs ---
        # چۆن (how), کوا (where), کەی (when), بۆچی (why)
        if token in _ADVERBIAL_QUESTION_WORDS:
            features.pos = "ADV"
            features.lemma = token
            return True
        
        # --- Adverb derivation detection [F#334, Haji Marf] ---
        # ADVERB_DERIVATION_PATTERNS: 4 source→affix mappings for
        # derived adverbs (بە+adj, adj+انە, noun+انە, بە+noun).
        # Prefix بە- is distinctive enough for POS assignment.
        # Suffix-based patterns (انە, یی) overlap with nominal morphology
        # (e.g. خانە in قوتابخانە), so they are tagged non-blocking.
        if not features.pos:
            for _src, affixes in ADVERB_DERIVATION_PATTERNS.items():
                for afx in affixes:
                    if afx == "بە" and token.startswith(afx) and len(token) > 3:
                        features.pos = "ADV"
                        features.lemma = token[len(afx):]
                        features.raw_analysis["adverb_derivation"] = _src
                        return True

        # --- Invariant adjectives (known list) ---
        if token in INVARIANT_ADJECTIVES:
            features.pos = "ADJ"
            features.lemma = token
            return True
        
        # --- Adjective by comparative/superlative suffix ---
        # ترین (superlative), تر (comparative) — Ibrahim (1988), pp. 66-67
        # F#316 (Haji Marf): 5 phonological assimilation rules apply when
        # stripping تر/ترین (COMPARATIVE_ASSIMILATION_RULES).
        if token.endswith("ترین") and len(token) > 4:
            features.pos = "ADJ"
            lemma = token[:-4]  # strip ترین
            # F#316: ت-final stems lose one ت before تر(ین)
            # e.g. کورت→کورتر (not *کورتتر); restore duplicated ت for lemma
            if lemma.endswith("ت") and not token[:-4].endswith("تت"):
                pass  # no duplication needed — natural ت stem
            features.lemma = lemma
            return True
        if token.endswith("تر") and len(token) > 2:
            features.pos = "ADJ"
            lemma = token[:-2]  # strip تر
            features.lemma = lemma
            return True
        
        # --- هەرە+ superlative prefix ---
        # F#319 (Haji Marf): هەرە can appear post-nominally too
        if token.startswith("هەرە") and len(token) > 4:
            features.pos = "ADJ"
            features.lemma = token[4:]  # strip هەرە
            return True

        # --- Adjective diminutive suffix detection moved to
        # _extract_nominal_features as non-blocking tag (F#323) ---

        # --- Demonstrative+preposition contractions [F#123, Haji Marf 2014] ---
        # بەم, لەم, بەو, لەو etc. are contracted DET forms
        _CONTRACTED_FORMS = {v: k for k, v in DEMONSTRATIVE_PREPOSITION_CONTRACTIONS.items()}
        if token in _CONTRACTED_FORMS:
            features.pos = "DET"
            features.lemma = token
            return True

        # --- Ordinal number suffixes [F#324, Haji Marf] ---
        # ORDINAL_SUFFIX_SHORT="ەم", ORDINAL_SUFFIX_LONG="ەمین" (F#328)
        # COMPOUND_ORDINAL_LAST_ONLY (F#329): in compound ordinals,
        # only the last numeral takes the ordinal suffix.
        for osuf in ORDINAL_SUFFIXES:
            if token.endswith(osuf) and len(token) > len(osuf) + 1:
                base = token[:-len(osuf)]
                if base in KURDISH_NUMERAL_WORDS or base.isdigit():
                    features.pos = "NUM"
                    features.lemma = base
                    # Tag which ordinal allomorph was used
                    if osuf == ORDINAL_SUFFIX_SHORT:
                        features.raw_analysis["ordinal_form"] = "short"
                    elif osuf == ORDINAL_SUFFIX_LONG:
                        features.raw_analysis["ordinal_form"] = "long"
                    return True

        # --- Fractional numbers [F#330, Haji Marf] ---
        # F#333: DISTRIBUTIVE_IS_ADVERB — distributive reduplication forms
        # (یەک یەک, دوو دوو) are adverbs, not numbers; these are multi-
        # token patterns detected at the sentence level, not here.
        if token in FRACTIONAL_NUMBER_PATTERNS:
            features.pos = "NUM"
            features.lemma = token
            return True

        # --- نیو/نیوە abstract vs concrete half [F#340, Haji Marf] ---
        # NIWE_ABSTRACT_CONCRETE distinguishes نیو (abstract, e.g. نیو شەو)
        # from نیوە (concrete, e.g. نیوەی سێوەکەم).
        if token in NIWE_ABSTRACT_CONCRETE:
            features.pos = "NUM"
            features.lemma = token
            features.raw_analysis["niwe_type"] = NIWE_ABSTRACT_CONCRETE[token]
            return True

        return False
    
    def _extract_verb_features(self, token: str, features: MorphFeatures) -> None:
        """Rule-based verb morpheme slot decomposition.
        
        Based on the morpheme slot chart in Amin (2016), p. 62:
            [negation][compound][mood/aspect] STEM [voice][person/number]
        
        Extended with:
        - Passive morpheme detection ({RA}/{ڕێ}) — Farhadi (2013), Finding #56
        - Past morpheme allomorph identification — Rule R6
        - Irregular present stem mapping — Finding #17
        - ش→ژ alternation — Rule R19
        """
        remaining = token

        # 0. Infinitive check — must come before prefix stripping because
        #    tokens like بردن start with ب which would be parsed as
        #    subjunctive prefix. Infinitive = past_stem + allomorph + ن.
        #    Source: Amin (2016), pp. 144-175
        #    INTRANSITIVE_DOUBLET_PAIRS (F#354): 10 pairs of variant
        #    infinitive forms (e.g. لەرزان/لەرزین) — accept both.
        if remaining.endswith(INFINITIVE_SUFFIX) and len(remaining) > 2:
            stem_candidate = remaining[:-1]  # strip final ن
            found_infinitive = False
            for allomorph, stems in PAST_MORPHEME_ALLOMORPHS.items():
                if stem_candidate.endswith(allomorph):
                    root = stem_candidate[:-len(allomorph)] if allomorph else stem_candidate
                    if root in stems:
                        found_infinitive = True
                        break
            # Also check irregular infinitives: past_stem + ن
            if not found_infinitive:
                for past_stem in IRREGULAR_PRESENT_STEMS:
                    if stem_candidate == past_stem:
                        found_infinitive = True
                        break
            if found_infinitive:
                features.tense = "infinitive"
                features.pos = "VERB"
                features.lemma = remaining
                # F#354: Check if this infinitive has a doublet variant
                for pair_a, pair_b in INTRANSITIVE_DOUBLET_PAIRS:
                    if remaining == pair_a or remaining == pair_b:
                        features.raw_analysis["has_doublet_variant"] = True
                        break
                return
        
        # 1. Check negation prefix
        # NEGATION_SLOT_MORPHEMES (F#342): نا=past, نە=present/subj, مە=imperative
        for neg in _NEGATION_PREFIXES_SORTED:
            if remaining.startswith(neg):
                features.negated = True
                remaining = remaining[len(neg):]
                # مە- indicates imperative (matches NEGATION_SLOT_MORPHEMES["مە"])
                if neg == "مە":
                    features.tense = "imperative"
                # Store which negation slot morpheme matched
                features.raw_analysis["negation_morpheme"] = neg
                break
        
        # 2. Check compound verb prefix
        for cpx in _COMPOUND_PREFIXES_SORTED:
            if remaining.startswith(cpx):
                features.compound_prefix = cpx
                remaining = remaining[len(cpx):]
                break

        # 2b. Obligatory preverb root validation [F#357, Haji Marf]
        # Some roots (پڕووزان, پشکنین) MUST have a preverb — bare use
        # is ungrammatical. Flag if no compound prefix was detected.
        if not features.compound_prefix:
            for obl_root in OBLIGATORY_PREVERB_ROOTS:
                if remaining.startswith(obl_root) or token.startswith(obl_root):
                    features.raw_analysis["missing_obligatory_preverb"] = obl_root
                    break
        
        # 3. Check mood/aspect prefix → determines tense
        has_mood_prefix = False
        for mood in _MOOD_ASPECT_PREFIXES_SORTED:
            if remaining.startswith(mood):
                if mood == "دە":
                    if not features.tense:
                        features.tense = "present"
                    features.aspect = "habitual"
                    has_mood_prefix = True
                elif mood == "ئە":
                    features.tense = "future"
                    has_mood_prefix = True
                elif mood == "ب":
                    # ب is highly ambiguous (many nouns start with it).
                    # Strengthened ensemble heuristic: require multiple
                    # evidence signals to accept ب as subjunctive prefix.
                    # Single-signal acceptance only for negation/compound
                    # (which are themselves strong verb indicators).
                    stem_after_b = remaining[1:]
                    evidence_score = self._score_b_prefix_evidence(
                        stem_after_b, features.negated, features.compound_prefix,
                    )
                    
                    # Require score >= 2: either negation/compound (auto-2),
                    # or known present stem (2), or stem_match + suffix (1+1),
                    # or suffix + allomorph core (1+1)
                    if evidence_score >= 2:
                        if not features.tense:
                            features.tense = "imperative"
                        has_mood_prefix = True
                    else:
                        break  # don't strip ب prefix
                
                if has_mood_prefix:
                    remaining = remaining[len(mood):]
                    features.pos = features.pos or "VERB"
                break
        
        # 3b. Detect epenthetic ت (Finding #167)
        # Source: Rasul (2005), pp. 28-29
        # Vowel-final verb stems insert non-morphemic ت before vowel
        # suffixes. Detect and flag to prevent misanalysis of ت as a
        # person marker.  First checks known stems, then falls back to
        # a vowel+ت+vowel pattern match for unknown stems.
        if not features.raw_analysis.get("has_epenthetic_t"):
            for epi_stem in EPENTHETIC_T_VERB_STEMS:
                for left_v, right_v in EPENTHETIC_T_ENVIRONMENTS:
                    if not epi_stem.endswith(left_v):
                        continue
                    epi_form = epi_stem + "ت" + right_v
                    if remaining == epi_form or remaining.startswith(epi_form):
                        features.raw_analysis["has_epenthetic_t"] = True
                        features.pos = features.pos or "VERB"
                        features.tense = features.tense or "present"
                        features.person = features.person or "3"
                        features.number = features.number or "sg"
                        if not features.lemma:
                            features.lemma = epi_stem
                        break
                if features.raw_analysis.get("has_epenthetic_t"):
                    break
            # Pattern fallback: vowel + ت + right_vowel for ALL environments
            # Catches rare stems not in EPENTHETIC_T_VERB_STEMS.
            # Anchored to remaining end to avoid substring false positives.
            # Runs when verb evidence exists (mood prefix, negation, or
            # compound prefix) to avoid false positives on nouns.
            if (not features.raw_analysis.get("has_epenthetic_t")
                    and (has_mood_prefix or features.negated
                         or features.compound_prefix)):
                for left_v, right_v in EPENTHETIC_T_ENVIRONMENTS:
                    pat = left_v + "ت" + right_v
                    if remaining.endswith(pat) and len(remaining) > len(pat):
                        features.raw_analysis["has_epenthetic_t"] = True
                        features.pos = features.pos or "VERB"
                        break
        
        # 4. If no mood/aspect prefix found, check if remaining is a known
        #    past stem — past tense has no prefix in Sorani (Amin 2016, p. 51)
        past_stem_len = 0  # length of matched stem+allomorph in remaining
        if not features.tense and not features.pos:
            for allomorph, stem_set in PAST_MORPHEME_ALLOMORPHS.items():
                for stem in stem_set:
                    full_past = stem + allomorph
                    if remaining.startswith(full_past):
                        features.tense = "past"
                        features.pos = "VERB"
                        past_stem_len = len(full_past)
                        break
                if features.tense == "past":
                    break
            # Also check irregular stems
            if not features.tense:
                for past_stem in IRREGULAR_PRESENT_STEMS:
                    if remaining.startswith(past_stem):
                        features.tense = "past"
                        features.pos = "VERB"
                        past_stem_len = len(past_stem)
                        break
        
        # 4b. Ahmadi Lexical Data match for Transitivity and POS
        if self._lexicon is not None and self._lexicon.available:
            try:
                match = self._lexicon.find_verb_stem(remaining)
            except Exception:
                logger.warning("Lexicon find_verb_stem() failed for: %.50s", remaining)
                match = None
            if match is not None:
                matched_stem, entry = match
                features.pos = features.pos or "VERB"
                
                # Finding #80: Transitivity extraction for Split Ergativity
                is_trans = ("T" in entry.flags or "M" in entry.flags)
                is_intrans = ("I" in entry.flags or "V" in entry.flags)
                
                features.raw_analysis["ahmadi_transitive"] = is_trans
                features.raw_analysis["ahmadi_intransitive"] = is_intrans
                
                if is_trans:
                    features.transitivity = "trans"
                elif is_intrans:
                    features.transitivity = "intrans"
                    
                if entry.lemma and not features.lemma:
                    features.lemma = entry.lemma
                if entry.inflectional and not features.tense:
                    if "past" in entry.inflectional:
                        features.tense = "past"
                    elif "present" in entry.inflectional:
                        features.tense = "present"

        # 5. Check voice suffixes (passive/causative) before person suffix
        #    Voice suffix may precede person suffix in the word, so search
        #    near the end rather than requiring endswith.
        _MAX_PERSON_SUFFIX_LEN = 3  # longest person suffix (e.g. ێت, ین)
        for voice_suf in _VOICE_SUFFIXES_SORTED:
            idx = remaining.rfind(voice_suf)
            # Must have a non-empty stem before it, and be near the end
            # (within person-suffix distance)
            if idx > 0 and len(remaining) - (idx + len(voice_suf)) <= _MAX_PERSON_SUFFIX_LEN:
                if voice_suf in ("ڕا", "ڕێ"):
                    features.voice = "passive"
                elif voice_suf in ("اند", "ێن"):
                    features.voice = "causative"
                features.pos = features.pos or "VERB"
                break

        # 5b. Causative suffix ئەوە [F#36, Haji Marf 2014]
        # CAUSATIVE_SUFFIX_EWE marks a productive causative derivation.
        # F#372: EWE_SUFFIX_FUNCTIONS — -ەوە serves 4 semantic functions:
        # repetition, meaning change, return to origin, concept change.
        # CAUSATIVE_A_TO_E_MAP captures stem-vowel changes (ا→ێ).
        # CAUSATIVE_BANNED_VERBS (F#347): 5 verbs that cannot form causatives.
        _causative_suffixes = tuple(CAUSATIVE_SUFFIX_EWE)
        if not features.voice and remaining.endswith(_causative_suffixes):
            # Find which suffix matched
            matched_suf = next(s for s in _causative_suffixes if remaining.endswith(s))
            stem_before = remaining[:-len(matched_suf)]
            # F#347: Skip causative detection for banned verb stems
            is_banned = any(
                stem_before.endswith(root[:-1]) or stem_before == root[:-1]
                for root in CAUSATIVE_BANNED_VERBS
            )
            if len(stem_before) >= 2 and not is_banned:
                features.voice = "causative"
                features.pos = features.pos or "VERB"
                # Check for vowel alternation in the causative stem
                for base_vowel, caus_vowel in CAUSATIVE_A_TO_E_MAP.items():
                    if caus_vowel in stem_before:
                        features.raw_analysis["causative_vowel_shift"] = True
                        break
        
        # 6. Check person/number suffix (present set, longest first)
        #    Only assign VERB from suffix alone if we already have a verb
        #    indicator (prefix, negation, known stem) — otherwise ان/م/ت
        #    suffixes falsely match nouns (کتێبەکان, پارەکەم).
        has_verb_evidence = features.pos == "VERB" or features.negated or bool(features.tense)
        
        # Exception: Finding #109 Portmanteau 3sg transitive present perfect ('ەتی' / 'یتی')
        portmanteau_match = False
        for pm_suf in (PORTMANTEAU_3SG_TRANS_PERFECT, PORTMANTEAU_3SG_TRANS_PERFECT_ALT):
            if remaining.endswith(pm_suf) and len(remaining) > len(pm_suf):
                portmanteau_match = True
                break
        if portmanteau_match:
            features.person = "3"
            features.number = "sg"
            features.tense = "present"
            features.transitivity = "trans"
            features.pos = features.pos or "VERB"
            features.raw_analysis["is_portmanteau_perfect"] = True
        else:
            for suffix in _PERSON_SUFFIXES_SORTED:
                if remaining.endswith(suffix) and len(remaining) > len(suffix):
                    # Don't let the person suffix consume the past-stem
                    # allomorph (e.g. ت in ڕۆیشت = ڕۆیش + allomorph ت)
                    if past_stem_len and len(remaining) - len(suffix) < past_stem_len:
                        break
                    pn = self._suffix_to_person_number(suffix)
                    if pn:
                        features.person, features.number = pn
                        if has_verb_evidence:
                            features.pos = features.pos or "VERB"
                    break

        # 7. Aspect inference — Qader (2017), "Tense Morphemes in Kurdish"
        # دە- already sets aspect="habitual" in step 3 above.
        # -وو-/-ووە infix → perfect (resultative completion).
        # Past tense without دە- prefix → perfective (punctual).
        if features.pos == "VERB" and not features.aspect:
            if "وو" in remaining or token.endswith("ووە"):
                features.aspect = "perfect"
            elif features.tense == "past":
                features.aspect = "perfective"

        # 7b. Passive vowel change detection (F#367, Haji Marf)
        # PASSIVE_VOWEL_CHANGES maps verbs with irregular passive stems
        # (e.g. بردن→بران with vowel ە→ا). Validate passive morphology.
        if features.voice == "passive" and features.pos == "VERB":
            for _inf, (pres_root, passive_stem) in PASSIVE_VOWEL_CHANGES.items():
                if passive_stem in remaining or remaining.startswith(passive_stem[:3]):
                    features.raw_analysis["passive_vowel_change"] = True
                    break

        # 7c. Past stem final sound classification (F#364, Haji Marf)
        # PAST_STEM_FINAL_SOUNDS: 5 possible final phonemes after past-stem
        # formation (ت/د/ا/ی/وو). Tag for downstream allomorph prediction.
        if features.tense == "past" and features.pos == "VERB":
            for final_sound in PAST_STEM_FINAL_SOUNDS:
                if remaining.endswith(final_sound) or token.rstrip("متینان").endswith(final_sound):
                    features.raw_analysis["past_stem_final"] = final_sound
                    break

        # 7d. Present stem consonant mutation detection (F#370, Haji Marf)
        # 5 mutation patterns (ش→ژ, س→ز, etc.) for present-stem formation.
        if features.tense in ("present", "future") and features.pos == "VERB":
            for _name, rule in PRESENT_STEM_CONSONANT_MUTATIONS.items():
                if rule["to"] in remaining:
                    features.raw_analysis["present_stem_mutation"] = _name
                    break

        # Check Wistin (ویستن) exception: it uses Set 1 for the subject in both tenses.
        # Must match the stem itself, not as a substring of another word.
        # Check both the original token and remaining, since features.lemma
        # may not be set yet at this point in the pipeline.
        wistin_stems = {"ویست", "ویستن"}
        wistin_present = {"ەوێ", "دەوێ", "ناوێ", "بوێ"}
        wistin_present_prefixed = {"دەوێ", "ناوێ", "بوێ"}
        if features.lemma in wistin_stems or features.lemma in wistin_present:
            features.raw_analysis["is_wistin_exception"] = True
        elif remaining in wistin_stems or remaining in wistin_present:
            features.raw_analysis["is_wistin_exception"] = True
        elif any(token.startswith(pfx) for pfx in wistin_present_prefixed):
            features.raw_analysis["is_wistin_exception"] = True
        elif any(
            token.startswith(stem) or remaining.startswith(stem)
            for stem in wistin_stems
        ):
            features.raw_analysis["is_wistin_exception"] = True

        # 8. هەبوون negation form detection [F#348, Haji Marf]
        # HABUN_POSSESSION_NEGATION="نیە" (possessive non-existence)
        # HABUN_EXISTENCE_NEGATION="نییە" (existential non-existence)
        if token == HABUN_POSSESSION_NEGATION or token == HABUN_EXISTENCE_NEGATION:
            features.pos = "VERB"
            features.tense = "present"
            features.person = "3"
            features.number = "sg"
            features.negated = True
            if token == HABUN_POSSESSION_NEGATION:
                features.raw_analysis["habun_negation"] = "possession"
            else:
                features.raw_analysis["habun_negation"] = "existence"
    
    def _extract_nominal_features(self, token: str, features: MorphFeatures) -> None:
        """Rule-based noun/adjective feature extraction.
        
        Strips nominal suffixes (definiteness, plurality, ezafe) to identify
        the lemma and assign features. Based on:
        - Fatah & Qadir (2006), pp. 24-25 (morpheme segmentation)
        - Slevanayi (2001), pp. 47-48 (definiteness/plurality)
        - F#59: vowel-final stems → یان for plural
        - F#60: vowel-final stems → یەکە for definite
        - F#134: ەکە never on conjugated verb
        - F#165: ی/یی ezafe six scenarios
        """
        remaining = token
        
        # 1. Strip nominal suffixes (longest first)
        # INDEFINITE_ARTICLE_FORMS (F#272): 4 phonological allomorphs
        # (ێک, ەک, یەک, انێک) for indefinite article selection.
        # IZAFE_MARKERS (F#273): ی (possessive/genitive/attributive)
        # vs ە (attributive only per F#291).
        for suffix, feat_name, feat_val in NOMINAL_SUFFIXES_ORDERED:
            if remaining.endswith(suffix) and len(remaining) > len(suffix):
                setattr(features, feat_name, feat_val)
                remaining = remaining[:-len(suffix)]
                features.pos = features.pos or "NOUN"
                # Definite+plural combined suffixes
                if suffix in ("ەکان", "یەکان"):
                    features.number = "pl"
                break
        
        # 2. Check for ezafe suffix (linking particle)
        # ی after consonant, یی after vowel — F#165
        # MORPHEME_E_FUNCTIONS (F#244, Haji Marf): morpheme ئە serves
        # as izafe allomorph, demonstrative suffix, and vocative particle.
        if not features.pos:
            match = EZAFE_PATTERN.match(remaining)
            if match and len(match.group(1)) > 1:
                features.case = "ez"
                remaining = match.group(1)
                features.pos = features.pos or "NOUN"
        
        # 3. Noun derivation suffix detection [F#262, Haji Marf]
        # If a word ends with a known noun-derivation suffix, confirm NOUN.
        if not features.pos:
            for _cat, suffixes in NOUN_DERIVATION_SUFFIXES.items():
                for suf in suffixes:
                    if remaining.endswith(suf) and len(remaining) > len(suf) + 1:
                        features.pos = "NOUN"
                        features.lemma = remaining[:-len(suf)]
                        features.number = features.number or "sg"
                        break
                if features.pos:
                    break

        # 3b. Noun-deriving prefix detection [F#293, Haji Marf]
        if not features.pos:
            for pfx in NOUN_DERIVING_PREFIXES:
                if remaining.startswith(pfx) and len(remaining) > len(pfx) + 1:
                    features.pos = "NOUN"
                    features.number = features.number or "sg"
                    break

        # 3c. Diminutive noun suffix detection [F#281, Haji Marf]
        # DIMINUTIVE_NOUN_SUFFIXES: 11 diminutive forms (ڵە, لە, ۆکە, etc.)
        if not features.pos:
            for dim_suf in DIMINUTIVE_NOUN_SUFFIXES:
                if remaining.endswith(dim_suf) and len(remaining) > len(dim_suf) + 1:
                    features.pos = "NOUN"
                    features.lemma = remaining[:-len(dim_suf)]
                    features.number = features.number or "sg"
                    features.raw_analysis["is_diminutive"] = True
                    break

        # 3c2. Adjective diminutive suffix tag [F#323, Haji Marf]
        # ADJECTIVE_DIMINUTIVE_SUFFIXES: 14 forms. Non-blocking annotation
        # — tags possible diminutive adjectives without overriding POS,
        # since these suffixes overlap with nominal morphology.
        if features.pos == "ADJ" or not features.pos:
            for adj_dim in ADJECTIVE_DIMINUTIVE_SUFFIXES:
                if len(adj_dim) >= 2 and remaining.endswith(adj_dim) and len(remaining) > len(adj_dim) + 2:
                    features.raw_analysis["possible_diminutive_adj"] = True
                    break

        # 3c3. Adverb derivation suffix tag [F#334, Haji Marf]
        # Non-blocking: suffix patterns (انە, یی) overlap with nominal
        # morphology, so we tag rather than assign POS.
        if not features.raw_analysis.get("adverb_derivation"):
            for _src, affixes in ADVERB_DERIVATION_PATTERNS.items():
                for afx in affixes:
                    if afx != "بە" and remaining.endswith(afx) and len(remaining) > len(afx) + 2:
                        features.raw_analysis["possible_adverb_derivation"] = _src
                        break
                if features.raw_analysis.get("possible_adverb_derivation"):
                    break

        # 3d. Agent noun suffix detection [F#374, Haji Marf]
        # AGENT_NOUN_SUFFIX_CLASSES: 9 suffixes in 3 classes by transitivity.
        if not features.pos:
            for _cls, info in AGENT_NOUN_SUFFIX_CLASSES.items():
                for ag_suf in info["suffixes"]:
                    if remaining.endswith(ag_suf) and len(remaining) > len(ag_suf) + 1:
                        features.pos = "NOUN"
                        features.lemma = remaining[:-len(ag_suf)]
                        features.number = features.number or "sg"
                        features.raw_analysis["is_agent_noun"] = True
                        features.raw_analysis["agent_noun_class"] = _cls
                        break
                if features.pos:
                    break

        # 3e. Instrument noun prefix detection [F#376, Haji Marf]
        # INSTRUMENT_NOUN_PREFIX = "پێ" — obligatory prefix for instrument nouns.
        if not features.pos and remaining.startswith(INSTRUMENT_NOUN_PREFIX) and len(remaining) > 3:
            features.raw_analysis["possible_instrument_noun"] = True

        # 4. Set default number if not detected
        if features.pos in ("NOUN", "") and not features.number:
            features.number = features.number or "sg"
        
        # 5. Set lemma if not already set by KLPT
        if not features.lemma:
            features.lemma = remaining
    
    def _detect_clitic(self, token: str, features: MorphFeatures) -> None:
        """Detect if token hosts a pronominal clitic.
        
        Source: Amin (2016), pp. 17-18 — clitics: م/ت/ی/مان/تان/یان
        Haji Marif (2014) classifies these as personal/bound pronouns
        (جیناوی کەسی لکاو) and distinguishes first-set (agent) from
        second-set (patient) roles in ergative past-tense constructions.
        Finding #9: two clitic sets with distinct person/number mapping.
        
        POS-aware disambiguation for ی (3sg):
        - On VERB hosts: ی is a clitic (3sg agreement/patient)
        - On NOUN hosts with ezafe case: ی is ezafe, not clitic
        - On NOUN hosts without ezafe: ی is possessive clitic
        Source: Rasul (2004), pp. 126-127 (F#165)
        """
        for cl, (person, number) in CLITIC_PERSON_MAP.items():
            if token.endswith(cl) and len(token) > len(cl) + 1:
                # ی disambiguation: if host is nominal and has ezafe case,
                # the trailing ی is the ezafe linker, not a 3sg clitic.
                if cl == "ی" and features.case == "ez":
                    continue
                # On NOUN/ADJ hosts, trailing ی is more likely possessive
                # than clitic unless there's verb evidence nearby.
                # Mark it but flag for downstream disambiguation.
                if cl == "ی" and features.pos in ("NOUN", "ADJ", ""):
                    # F#283 (SUFFIX_YI_FUNCTIONS): ی has 3 distinct functions:
                    # izafe linker, feminine marker, or attribution particle.
                    features.raw_analysis["yi_ambiguous"] = True
                    features.raw_analysis["yi_functions"] = list(SUFFIX_YI_FUNCTIONS.keys())
                # F#122 (YISH_BEFORE_CLITIC_ONLY): ش/یش only appears
                # immediately before a clitic, never standalone.
                if YISH_BEFORE_CLITIC_ONLY and cl != "ی":
                    host = token[:-len(cl)]
                    if host.endswith("یش") or host.endswith("ش"):
                        features.raw_analysis["has_yish_preclitic"] = True
                features.is_clitic = True
                features.clitic_person = person
                features.clitic_number = number
                # Store clitic info in raw_analysis for downstream use
                features.raw_analysis["hosted_clitic"] = cl
                features.raw_analysis["clitic_person"] = person
                features.raw_analysis["clitic_number"] = number
                break
    
    @staticmethod
    def _suffix_to_person_number(suffix: str) -> tuple[str, str] | None:
        """Map a verb suffix to (person, number) tuple."""
        return _SUFFIX_PERSON_NUMBER.get(suffix)

    @staticmethod
    def _score_b_prefix_evidence(
        stem_after_b: str, negated: bool, compound_prefix: str,
    ) -> int:
        """Score evidence that ب is a subjunctive/imperative prefix.

        Returns an integer score; threshold of >= 2 indicates ب is
        likely a verb prefix rather than part of a noun stem.

        Evidence signals:
          - Negation/compound prefix already found → 2 (strong)
          - Known present stem → 2
          - Past allomorph stem → 1
          - ش→ژ alternation stem → 1
          - Person suffix + stem core match → 1-2
        """
        if negated or compound_prefix:
            return 2
        score = 0
        if any(stem_after_b.startswith(s)
               for s in _ALL_PRESENT_STEMS):
            score += 2
        elif any(stem_after_b.startswith(s)
                 for s in _ALL_PAST_STEMS):
            score += 1
        if any(stem_after_b.startswith(s)
               for s in _ALL_SH_ZH_STEMS):
            score += 1
        for suf in _PRESENT_SUFFIXES_SORTED:
            if (stem_after_b.endswith(suf)
                    and len(stem_after_b) > len(suf)
                    and len(stem_after_b) - len(suf) >= 2):
                score += 1
                stem_core = stem_after_b[:-len(suf)]
                if stem_core in _ALL_PAST_STEMS or stem_core in _ALL_PRESENT_STEMS:
                    score += 1
                break
        return score
    
    def analyze_sentence(self, sentence: str) -> list[MorphFeatures]:
        """Analyze all tokens in a sentence.
        
        After per-token analysis, applies sentence-level post-processing:
        - Oblique case marking on nouns/pronouns after prepositions
          (Sorani has no suffixal oblique; case is positional)
        """
        tokens = self.tokenize(sentence)
        features_list = [self.analyze_token(tok) for tok in tokens]

        # Post-processing: mark oblique case on complements of prepositions.
        # Source: Abbas & Sabir (2020) — preposition governs oblique on its
        # complement; Sorani marks this positionally, not morphologically.
        for i, feat in enumerate(features_list):
            if feat.pos == "ADP" and i + 1 < len(features_list):
                nxt = features_list[i + 1]
                if nxt.pos in ("NOUN", "PRON", "ADJ", "") and not nxt.case:
                    nxt.case = "obl"

        # Post-processing: cardinal numeral forces singular on following noun.
        # F#327 (Haji Marf): nouns after cardinal numbers stay singular.
        if NOUN_SINGULAR_AFTER_CARDINAL:
            for i, feat in enumerate(features_list):
                if feat.pos == "NUM" and i + 1 < len(features_list):
                    nxt = features_list[i + 1]
                    if nxt.pos in ("NOUN", "") and nxt.number == "pl":
                        nxt.number = "sg"
                        nxt.raw_analysis["cardinal_forced_sg"] = True

        return features_list
    
    def tokenize(self, text: str) -> list[str]:
        """Tokenize Sorani Kurdish text.
        
        Handles Arabic script punctuation, zero-width characters, and
        Kurdish-specific orthographic conventions.
        """
        if self._tokenize is not None:
            try:
                result = self._tokenize.word_tokenize(text)
                if result:
                    return result
            except (ValueError, AttributeError, TypeError):
                pass
        
        # Delegate to the shared tokenizer (src/data/tokenize.py) which
        # applies ZWJ removal, conjunctive و splitting, and regex-based
        # token extraction.
        return _sorani_tokenize(text)
    
    def build_feature_vocabulary(self) -> dict[str, int]:
        """Build vocabulary mapping for morphological features."""
        vocab = {"PAD": 0, "UNK": 1}
        idx = 2
        
        feature_values = {
            "person": ["1", "2", "3"],
            "number": ["sg", "pl"],
            "tense": ["present", "past", "future", "imperative", "infinitive"],
            "aspect": ["habitual", "perfective", "perfect"],
            "case": ["nom", "obl", "ez"],
            "definiteness": ["def", "indef"],
            "transitivity": ["trans", "intrans"],
            "clitic_person": ["1", "2", "3"],
            "clitic_number": ["sg", "pl"],
        }
        
        for feat_name, values in feature_values.items():
            for val in values:
                key = f"{feat_name}:{val}"
                vocab[key] = idx
                idx += 1
            vocab[f"{feat_name}:UNK"] = idx
            idx += 1
        
        return vocab


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    analyzer = MorphologicalAnalyzer(use_klpt=False)
    
    test_sentences = [
        "من دەچم بۆ قوتابخانە",
        "ئەو کتێبە جوانەکەی خوێندەوە",
        "ئایا تۆ دوێنێ چوویت بۆ بازاڕ؟",
    ]
    
    for sent in test_sentences:
        print(f"\n>>> {sent}")
        tokens = analyzer.tokenize(sent)
        for tok in tokens:
            f = analyzer.analyze_token(tok)
            print(f"  {tok}: pos={f.pos}, lemma={f.lemma}, "
                  f"person={f.person}, number={f.number}, "
                  f"tense={f.tense}, def={f.definiteness}")
