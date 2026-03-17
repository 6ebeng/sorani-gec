"""
Tense-Agreement Error Generator (Split-Ergative Pattern)

Injects tense-agreement errors exploiting Sorani Kurdish's split-ergative
system [F#80]. In present/future tenses, the verb agrees with the subject
(nominative-accusative) [F#1, F#66 Law 1]. In past tenses with transitive
verbs, the verb agrees with the object (ergative) [F#14, F#66 Law 2].

The split-ergative agreement paradigms are based on Slevanayi (2001)
"Agreement in the Sorani Variety of the Kurdish Language" [F#64, F#66] and
the verb conjugation tables in Amin (2016) "Verb Grammar of the Kurdish
Language" [F#95, F#98].

Maaruf (2010, pp. 78-89) formalises 14 verb templates [F#110] showing the
exact morpheme ordering in each tense/mood combination. The key insight for
tense-agreement errors is the clitic reversal between present and past [F#22]:
    Present: Neg + Aspect + Stem + Agent                (Law 1)
    Past transitive: Agent + Neg + Stem + Patient       (Law 2)
    Past intransitive: Neg + Stem + Agent               (Law 1)
A tense-agreement error arises when a writer applies the wrong template
(e.g. adding a Law-1 suffix to a Law-2 verb form, or vice versa).

Amin (2016, pp. 39-43) documents passive constructions [F#107] where the
verb loses its agent role, causing Set 1 clitics to shift to Set 2 behaviour.
This means passive past verbs follow Law 1 (like intransitives), not Law 2.

Fatah (2006) characterises Kurdish as a "morphologically incomplete ergative"
language [F#29], reinforcing that the ergative pattern is partial and
tense-conditioned.

Key findings implemented:
  F#1   — Subject-verb agreement as primary domain
  F#9   — Three clitic sets
  F#13  — Agentive vs patientive intransitive (ذفراست / ڕوودان)
  F#14  — Past transitive double clitic structure
  F#22  — Clitic role switching by tense
  F#29  — Ergativity detail (morphologically incomplete)
  F#35  — ڕوودان (happening) verbs use ێ for ALL persons [F#35]
  F#40  — Present perfect aspect marker ە
  F#56  — Passive morpheme را/ران
  F#66  — Two agreement laws
  F#80  — Split ergativity (nominative in present, ergative in past)
  F#94  — Nominative case unifies both laws (Amin 2016)
  F#96  — 3SG ات allomorph for 8 verbs
  F#98  — 8-tense conjugation formulas
  F#101 — ویستن exception (Set 1 for subject in both tenses)
  F#102 — Passive morpheme decomposition (ر+ا)
  F#107 — Passive clitic reassignment
  F#109 — Portmanteau ەتی (3sg agent + 2sg patient)
  F#110 — Verb morpheme ordering (14 templates)
  F#113 — Modal force subjunctive (دەوێت + بـ obligatory)

This creates a unique error type: using present-tense agreement in a past-tense
context, or vice versa.

Examples:
- Correct (past transitive, ergative): "من ئەو-م دیت" (I saw him — verb agrees with 'him')
- Error (using present agreement):     "من ئەو-م دیتم" (wrong: added 1sg as if nominative)
"""

import re
from typing import Optional

from .base import BaseErrorGenerator


# Past tense markers / patterns
# Source: Amin (2016), pp. 15, 17-18, 51-52 — expanded with irregular stems
#         and suppletive pairs from Amin p. 37
#         Kurdish Academy grammar (2018), pp. 80–181; Rasul (2005), pp. 4–16
PAST_TENSE_MARKERS = [
    # Common regular stems
    "بوو",     # was
    "کرد",     # did
    "چوو",     # went
    "هات",     # came
    "گوت",     # said
    "دیت",     # saw
    "نووسی",   # wrote
    "خوێند",   # read/studied
    "کڕی",     # bought
    "فرۆشت",   # sold
    "برد",     # took / carried
    "خوارد",   # ate
    "دا",      # gave
    "گرت",     # took / caught
    "ناسی",    # knew / recognised
    "گەیشت",   # arrived / reached
    "مرد",     # died (intransitive)
    "کەوت",    # fell (intransitive)
    "نیشت",    # sat down
    # Irregular stems — Amin (2016), p. 15
    "نارد",    # sent (نارد ← نێردن)
    "شکاند",   # broke-trans (شکاندن)
    "ڕشت",     # poured (ڕشتن)
    "هێنا",    # brought (هێنان — suppletive)
    "تاشی",    # carved (تاشین)
    "پێچا",    # wrapped (پێچان)
    "چنی",     # wove (چنین)
    "پاراست",  # protected (پاراستن)
    "نووست",   # slept (intransitive)
    "ڕۆیشت",   # went (dialectal form)
    "خلیسکا",  # slipped (intransitive)
    "ئاوابوو",  # was destroyed (intransitive)
    "ڕووخا",   # collapsed (intransitive)
    "شەقا",    # cracked (intransitive)
    "هەستا",   # stood up (intransitive)
    "پەڕی",    # jumped (intransitive)
    "دانیشت",  # sat (intransitive)
    # Suppletive pairs — Amin (2016), p. 37
    "ئاورد",   # brought (dialectal: هێنان/ئاوردن)
    "ئەست",    # threw (dialectal)
    # Expanded from Kurdish Academy (2018) + Rasul (2005)
    "خست",     # put/dropped
    "بەست",    # tied
    "بیست",    # heard
    "کێڵا",    # weighed
    "پێوا",    # measured
    "سپارد",   # deposited
    "ژمارد",   # counted
    "بژارد",   # chose
    "شارد",    # hid
    "کوشت",    # killed
    "شوشت",    # washed
    "هاوشت",   # threw
    "بەخشی",   # forgave
    # Patientive intransitive stems (ڕوودان/happening)
    "سووتا",   # burned
    "پسا",     # withered
    "خنکا",    # drowned
    "ڕژا",     # spilled
    "شکا",     # broke (intrans)
    "دڕا",     # tore
    "بڕا",     # was cut
    "ترسا",    # feared
    "فەوتا",   # was lost
    # Agentive intransitive expanded
    "گەڕا",    # returned
    "فڕی",     # flew
    "وەستا",   # stopped
    "گریا",    # cried
    "سووڕا",   # rode/slid
    "هەڵسا",   # got up
    # Causative past stems (converts intr → trans)
    "سووتاند",  # made burn
    "وەستاند",  # made stop
    "خزاند",   # made slide
    "مراند",   # drove/killed off
    "تەقاند",   # detonated
    "ڕژاند",   # poured (trans)
    "کەواند",   # dropped/felled
    "نواند",    # put to sleep
    "فراند",   # made fly
    "بەزاند",   # melted (trans)
]

# Present tense prefixes
PRESENT_PREFIXES = ["دە", "ئە"]

# Negation prefixes that can appear on past verbs: نە-
# Source: Amin (2016), p. 51
PAST_NEGATION_PREFIXES = ["نە"]

# Past tense verb endings (agreement with object in transitive — ergative)
# Source: Amin (2016), pp. 17-18; Saliqanai (2001), pp. 60-61
PAST_ENDINGS = {
    "1sg": "م",
    "2sg": "ت",
    "3sg": "",      # zero morpheme in past 3sg
    "1pl": "مان",
    "2pl": "تان",
    "3pl": "یان",
}

# Present tense verb endings (agreement with subject — nominative)
# Source: Amin (2016), pp. 17-18, 21-22
PRESENT_ENDINGS = {
    "1sg": "م",
    "2sg": "یت",
    "3sg": "ێت",
    "1pl": "ین",
    "2pl": "ن",
    "3pl": "ن",
}

# Eight verbs whose 3SG present takes ات instead of ێت
# Source: Amin (2016), pp. 21-22 — Finding #96
# Rule: present roots ending in ۆ or ە take ات (ۆ→و before ات, ە deleted)
# These are the combined stems (root + first vowel of ات) used for matching
AT_ALLOMORPH_STEMS = {
    "با",     # بردن → present root بە + ات → بات
    "کا",     # کردن → present root کە + ات → کات
    "خا",     # خستن → present root خە + ات → خات
    "شوا",    # شوشتن → present root شۆ → شو + ات → شوات
    "پوا",    # present root ends in ۆ → پو + ات → پوات
    "خوا",    # خواردن → present root خۆ → خو + ات → خوات
    "گا",     # گەیشتن → present root گە + ات → گات
    "دا",     # دان → present root دە + ات → دات
}

# Exceptional verbs that show non-standard agreement (double-clitic pattern)
# Source: Amin (2016), pp. 51-52 — Finding #101
# ویستن uses Set 1 for subject in BOTH past and present tenses,
# unlike normal transitives which switch to Set 2 in present.
EXCEPTIONAL_VERBS = {
    "ویست": "want",   # past of ویستن — Set 1 subject in both tenses
    "وێ": "want",     # present root of ویستن
}

# Causative suffixes — Amin (2016), pp. 30-33
# Causative morphology changes transitivity (intransitive→transitive),
# which in turn changes agreement law from Law 1 to Law 2 in past tense.
CAUSATIVE_PRESENT_SUFFIX = "ێن"   # e.g. دەچکێنێت 'makes drip'
CAUSATIVE_PAST_SUFFIX = "اند"     # e.g. چکاند 'made drip'

# ڕوودان (happening/non-volitional) present stems
# Source: Kurdish Academy grammar (2018), pp. 80–106 (Finding #35)
# These verbs use ێ for ALL persons in present tense
# (not just 3sg like regular verbs)
# e.g., دەسووتێم, دەسووتێی, دەسووتێ, دەسووتێین, دەسووتێن
RUUDAN_PRESENT_STEMS = [
    "سووتێ",   # burn
    "پسێ",     # wither
    "خنکێ",    # drown
    "ڕژێ",     # spill
    "شکێ",     # break (intrans)
    "دڕێ",     # tear
    "بڕێ",     # cut (intrans)
    "ترسێ",    # fear
    "فەوتێ",   # be lost
    "خزێ",     # slide
    "تەقێ",    # explode
    "بزێ",     # vibrate
    "کۆکێ",    # boil
    "پژمێ",    # wilt
    "گەڕێ",    # return (happen.)
    "برژێ",    # fry
]

# Passive morphemes — Farhadi (2013), Finding #56; Amin (2016), pp. 39-44, Finding #102
# Passive always takes Set 2 clitics (same as intransitive)
# Amin decomposes: ر = passive morpheme, ا/ێ = tense markers
# For matching purposes we use the combined forms:
PASSIVE_PAST_MARKER = "ڕا"   # past passive: کوژرا = کوژ+ر+ا
PASSIVE_PRESENT_MARKER = "ڕێ"  # present passive: دەکوژرێت = دە+کوژ+ر+ێ+ت

# 5 verbs that form passive from PAST root instead of present root
# Source: Amin (2016), p. 41 — Finding #102
PASSIVE_PAST_ROOT_EXCEPTIONS = {
    "ویست",   # ویستن → ویستڕا / دەویستڕێ (not *وێڕا)
    "وت",     # وتن → وتڕا / دەوتڕێ (not *ڵێڕا)
    "گوت",    # گوتن → گوتڕا / دەگوتڕێ (not *ڵێڕا)
    "لیست",   # لیستن → لیستڕا / دەلیستڕێ (not *لێسڕا)
    "بیست",   # بیستن → بیستڕا / دەبیستڕێ (not *بیسڕا)
}

# Present perfect ە suffix — Kurdish Academy (2018), Finding #40, Rule R17
# In present perfect transitive, ە copula is obligatory on all persons
# e.g., گرتوومە, گرتووتە, گرتوویە
PRESENT_PERFECT_COPULA = "ە"

# ـەتی portmanteau for 3sg transitive present perfect — Amin (2016), p. 56
# Source: Finding #109
# Fused form: root + وو + ی(Set1 3sg) + ە(copula) + تی
# Example: بردوویەتی "he has taken it"
PRESENT_PERFECT_3SG_TRANS_PORTMANTEAU = "ەتی"

# Passive clitic reassignment table — Amin (2016), p. 43 — Finding #107
# When active → passive, Set 1 object clitics convert to Set 2 subject clitics:
PASSIVE_CLITIC_REASSIGNMENT = {
    "م": "م",        # 1sg: Set1 م → Set2 م (same form)
    "ت": "یت",       # 2sg: Set1 ت → Set2 یت
    "ی": "ێت",       # 3sg: Set1 ی → Set2 ێت
    "مان": "ین",     # 1pl: Set1 مان → Set2 ین
    "تان": "ن",      # 2pl: Set1 تان → Set2 ن
    "یان": "ن",      # 3pl: Set1 یان → Set2 ن
}

# Verbs that can NEVER be made transitive (no causative ێن/اند possible)
# Source: Amin (2016), pp. 36-37 — Finding #112
NEVER_TRANSITIVIZABLE_VERBS = {"گەنین", "زان", "پشکووتن", "زەپین"}

# Modal (هەوەس) verbs that force embedded verb into subjunctive بـ
# Source: Amin (2016), p. 45 — Finding #113
# The embedded main verb must use بـ prefix (subjunctive), NEVER دە- (indicative).
# Example: دەتوانین بڕۆین (correct) vs *دەتوانین دەڕۆین (error)
MODAL_VERBS = {
    "توان": "can",       # دەتوانین
    "بوا": "should",     # دەبوایە
    "شی": "may/can",     # دەشێ
}

# Compound verb preverbs — needed for imperative بـ optionality (Finding #111)
# In compound verbs with these preverbs, بـ is OPTIONAL in imperative.
# In negative imperative, مە completely replaces بـ.
# Source: Amin (2016), p. 35
IMPERATIVE_PREVERBS = ["تێ", "پێ", "لێ", "دا", "هەڵ", "دەر", "ڕا", "وەر"]

# Morpheme slot ordering — Amin (2016), pp. 55, 62 — Finding #110
# Definitive template: morph_prefix + infl_prefix + ROOT + tense_marker
#   + aspect/voice + agreement + morph_suffix
# This documents the valid orderings for morphological validation.
MORPHEME_SLOT_ORDER = [
    "morphological_prefix",   # تێ/دا/دەر/ڕا/لێ/هەڵ/وەر/پێ
    "inflectional_prefix",    # دە/بـ/نە/نا/مە/نی + Set1_clitic(when prefixal)
    "root",                   # verb root (present or past)
    "tense_marker",           # ا/ی/ت/د/وو/× (past) or $/ی̃ (present)
    "aspect_voice",           # بوو/وو+ە/ایە/ر+ا/ر+ێ/ێن/اند
    "agreement_suffix",       # Set2 clitic (when suffixal)
    "morphological_suffix",   # ەوە/ن
]


class TenseAgreementErrorGenerator(BaseErrorGenerator):
    """Generate tense-agreement errors exploiting split-ergative patterns."""
    
    @property
    def error_type(self) -> str:
        return "tense_agreement"
    
    def find_eligible_positions(self, sentence: str) -> list[dict]:
        """Find past-tense verbs where ergative agreement can be disrupted.
        
        Handles plain and negated past verbs (نەکرد, نەچوو, etc.)
        per Amin (2016), p. 51.
        """
        positions = []
        
        # Look for past tense verbs (common stems), including negated forms
        for past_verb in PAST_TENSE_MARKERS:
            # Match: optional negation prefix + optional compound prefix + past stem + optional suffixes
            neg_alt = "|".join(re.escape(p) for p in PAST_NEGATION_PREFIXES)
            pattern = re.compile(
                rf'\b((?:{neg_alt})?\w*{re.escape(past_verb)}\w*)\b'
            )
            
            for match in pattern.finditer(sentence):
                word = match.group(1)
                
                # Check it's not a present-tense verb (no present prefix)
                if any(word.startswith(pref) for pref in PRESENT_PREFIXES):
                    continue
                
                # Flag exceptional verbs for possible special handling
                is_exceptional = past_verb in EXCEPTIONAL_VERBS
                
                positions.append({
                    "start": match.start(),
                    "end": match.end(),
                    "original": word,
                    "context": {
                        "verb": word,
                        "base_stem": past_verb,
                        "is_exceptional": is_exceptional,
                    },
                })
        
        return positions
    
    def generate_error(self, position: dict) -> Optional[str]:
        """Add wrong agreement suffix to a past-tense verb.
        
        Strategy: Add a present-tense-style ending to a past form,
        creating a tense-agreement mismatch.
        """
        ctx = position["context"]
        verb = ctx["verb"]
        
        # Pick a random present-tense ending and append it
        pn = self.rng.choice(list(PRESENT_ENDINGS.keys()))
        ending = PRESENT_ENDINGS[pn]
        
        if not ending:
            return None
        
        # Add the incorrect ending
        error_verb = verb + ending
        
        if error_verb == verb:
            return None
        
        return error_verb
