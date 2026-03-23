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
    DEMONSTRATIVES,
    EPENTHETIC_T_ENVIRONMENTS,
    EPENTHETIC_T_VERB_STEMS,
    INTERROGATIVE_PRONOUNS,
    INVARIANT_ADJECTIVES,
    OPTATIVE_SENTENCE_PARTICLES,
    QUANTIFIER_FORMS,
    QUESTION_WORDS,
    RECIPROCAL_PRONOUNS,
    SUBJECT_PRONOUNS,
    YESNO_QUESTION_PARTICLES,
)

logger = logging.getLogger(__name__)

# Pre-compiled tokenizer regex — avoids re-compilation per call.
_TOKENIZE_PATTERN = re.compile(
    r'[\u0621-\u063A\u0641-\u064A\u066E-\u06D3\u06D5-\u06EF'
    r'\u06FA-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF'
    r'\u200C'
    r']+'
    r'|[\u06F0-\u06F90-9]+'
    r'|[a-zA-Z]+'
    r'|[^\s]'
)

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
# Source: Rasul (2005), p. 22; Kurdish Academy (2018), pp. 167–181
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
DEFINITE_SUFFIX = "ەکە"       # definiteness marker (the)
INDEFINITE_SUFFIX = "ێک"      # indefiniteness marker (a/an)
PLURAL_SUFFIX = "ان"          # plural marker
DEMONSTRATIVE_SUFFIX = "ە"    # proximal demonstrative

# Nominal suffix patterns (ordered longest-first for greedy match)
# Source: Fatah & Qadir (2006), pp. 24-25; Slevanayi (2001), pp. 47-48
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
            except Exception:
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
        
        # --- Invariant adjectives (known list) ---
        if token in INVARIANT_ADJECTIVES:
            features.pos = "ADJ"
            features.lemma = token
            return True
        
        # --- Adjective by comparative/superlative suffix ---
        # ترین (superlative), تر (comparative) — Ibrahim (1988), pp. 66-67
        if token.endswith("ترین") and len(token) > 4:
            features.pos = "ADJ"
            features.lemma = token[:-4]  # strip ترین
            return True
        if token.endswith("تر") and len(token) > 2:
            features.pos = "ADJ"
            features.lemma = token[:-2]  # strip تر
            return True
        
        # --- هەرە+ superlative prefix ---
        if token.startswith("هەرە") and len(token) > 4:
            features.pos = "ADJ"
            features.lemma = token[4:]  # strip هەرە
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
                return
        
        # 1. Check negation prefix
        for neg in _NEGATION_PREFIXES_SORTED:
            if remaining.startswith(neg):
                features.negated = True
                remaining = remaining[len(neg):]
                # مە- indicates imperative
                if neg == "مە":
                    features.tense = "imperative"
                break
        
        # 2. Check compound verb prefix
        for cpx in _COMPOUND_PREFIXES_SORTED:
            if remaining.startswith(cpx):
                features.compound_prefix = cpx
                remaining = remaining[len(cpx):]
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
            match = self._lexicon.find_verb_stem(remaining)
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
        if not features.pos:
            match = EZAFE_PATTERN.match(remaining)
            if match and len(match.group(1)) > 1:
                features.case = "ez"
                remaining = match.group(1)
                features.pos = features.pos or "NOUN"
        
        # 3. Set default number if not detected
        if features.pos in ("NOUN", "") and not features.number:
            features.number = features.number or "sg"
        
        # 4. Set lemma if not already set by KLPT
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
                    features.raw_analysis["yi_ambiguous"] = True
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
            except Exception:
                pass
        
        # Rule-based tokenizer: handles Kurdish text without external deps
        # 1. Normalize zero-width characters (ZWNJ, ZWJ) — keep ZWNJ as it
        #    has morphological significance in Kurdish (separates morphemes)
        text = text.replace('\u200d', '')  # remove ZWJ (zero-width joiner)
        
        # 1b. Split conjunctive و when attached to next word.
        # Sorani و ("and") often prefixes the following word without a space
        # (e.g. وئەو = و + ئەو). Split only when followed by a known
        # word-initial letter pattern, not when و is part of the stem.
        text = re.sub(
            r'(?:(?<=\s)|^)و(?=[ئابتجحخدذرزسشصضطظعغفقكلمنهکگڵیێەپڕژڤڶ])',
            'و ', text
        )
        
        # 2. Split on whitespace, separating Arabic punctuation from words.
        # Arabic punctuation (٬060C, ؛061B, ؟061F, ۔06D4) must NOT merge
        # with adjacent Arabic letters. Use explicit letter ranges.
        tokens = _TOKENIZE_PATTERN.findall(text)
        return [t for t in tokens if t.strip()]
    
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
