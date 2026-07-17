"""
Central Kurdish (Sorani) Agreement Accuracy Checker

Sorani-specific evaluation metric that checks whether agreement constraints
are satisfied in the model's output. Measures the percentage of sentences
where targeted agreement checks pass.

Uses the rule-based morphological analyzer (Amin 2016, Fatah & Qadir 2006)
and agreement constants from Slevanayi (2001) to detect violations.
"""

import logging
from dataclasses import dataclass
from typing import Optional

from ..morphology.analyzer import (
    MorphologicalAnalyzer,
    CLITIC_PERSON_MAP,
)
from ..morphology.constants import (
    SUBJECT_PRONOUNS,
    TRANSITIVE_PAST_STEMS,
    CLITIC_BARRED_PRONOUNS,
    RECIPROCAL_VARIANTS,
)
from ..morphology.builder import (
    _is_present_verb,
    _is_transitive_past,
    _is_past_verb,
    build_agreement_graph,
)

logger = logging.getLogger(__name__)

# Subject pronouns → expected verb person/number
# Canonical source: SUBJECT_PRONOUNS from constants.py (Amin 2016, pp. 17-18)
_PRONOUN_AGREEMENT = SUBJECT_PRONOUNS

# Present-tense verb ending → (person, number)
# These are Set 2 agreement suffixes on verbs (NOT Set 1 clitics).
# Used by _check_subject_verb() to extract verb person/number.
# Source: Amin (2016), pp. 17-18
_PRESENT_ENDINGS: dict[str, tuple[str, str]] = {
    "م":   ("1", "sg"),   # Set 2: 1sg
    "ەم":  ("1", "sg"),   # Set 2: 1sg (epenthetic)
    "یت":  ("2", "sg"),   # Set 2: 2sg
    "ێت":  ("3", "sg"),   # Set 2: 3sg
    "ات":  ("3", "sg"),   # Set 2: 3sg (after -a stems)
    "ێ":   ("3", "sg"),   # Set 2: 3sg (short form)
    "ین":  ("1", "pl"),   # Set 2: 1pl
    "ن":   ("3", "pl"),   # Set 2: 3pl
    "ەن":  ("3", "pl"),   # Set 2: 3pl (epenthetic)
}

# Verb prefixes that indicate tense
_PRESENT_PREFIXES = ("دە", "ئە")
_NEGATION_PRESENT_PREFIX = "نا"
_NEGATION_PAST_PREFIX = "نە"
_IMPERATIVE_PREFIX = "ب"

# Full demonstrative forms whose initial ئە drops after بە/لە (F#123)
_DEM_FULL_FORMS = ("ئەمانە", "ئەوانە", "ئەمە", "ئەوە", "ئەم", "ئەو")

# Proper place names for F#86 (Slevanayi 2001, pp. 43-44): proper nouns
# never take indefinite/plural markers. Names whose gentilic coincides
# with the bare name (سلێمانی) are excluded to avoid false positives.
_PROPER_PLACE_NAMES = (
    "هەولێر", "دهۆک", "کەرکووک", "زاخۆ", "هەڵەبجە", "کۆیە", "ڕانیە",
    "بەغدا", "کوردستان", "عێراق",
)

# High-frequency present stems that do NOT begin with ب — safe anchors for
# scanning after the نەب/مەب/مە prefixes (R14, F#157). ب-initial stems
# (بین, بڕ, بەس) are excluded because نە+بینم is a legitimate negated
# subjunctive, not a نە+ب co-occurrence error.
_PRESENT_STEMS_NON_B = (
    "نووس", "خوێن", "فرۆش", "کوژ", "نێر", "زان", "کڕ", "گر",
    "کەو", "خۆ", "ڕۆ", "دە", "کە", "چ",
)

# Consonant-final present stems whose 2sg imperative requires final ە
# (R15/F#42): بنووسە/بگرە, never bare *بنووس/*بگر. Patientive مر is
# excluded — its imperative is impossible altogether (F#13).
_IMPERATIVE_E_STEMS = ("نووس", "خوێن", "فرۆش", "کوژ", "نێر", "زان", "کڕ", "گر")

# Intransitive present stems for F#125 (Haji Marf 2014, p. 192): an
# intransitive imperative can never host a Set 1 clitic (*بمکەوە).
# نوو (sleep) is excluded — it is a prefix of transitive نووس (write).
_INTRANS_PRESENT_STEMS = ("کەو", "چ", "ڕۆ", "خەو")

# Present stems with the ش→ژ alternation (R19, Academy Committee):
# the infinitive's ش (کوشتن, هاوێشتن) becomes ژ in the present stem
# (دەکوژم, دەهاوێژم) — *دەکوشم keeps the past-stem consonant.
_SH_ZH_PAST_STEM_PREFIXES = ("کوش", "هاوێش")

# Past intransitive full forms = stem + Set 2 suffix (Academy Committee,
# pp. 144-156). وەستا is excluded (homograph of the noun "master"), ما
# is too short (مام "uncle").
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
# Contexts that can legitimately precede a SUBJECT pronoun. Anything
# else (prepositions, ezafe-linked possessors like براکەی من) blocks the
# past agreement check to avoid false positives.
_SUBJ_CONTEXT_OK = {
    "بەڵام", "کە", "چونکە", "ئەگەر", "کاتێک", "پاشان", "ئینجا",
    "دوێنێ", "ئەمڕۆ", "ئێستا", "بۆیە", "،", ".", "؟", "!",
}

# F#87 familiarity hierarchy (Slevanayi 2001, pp. 68-69): coordinated
# subjects order 1st > 2nd > 3rd person.
_FAMILIARITY_RANK = {
    "من": 1, "ئێمە": 1, "منیش": 1, "ئێمەش": 1,
    "تۆ": 2, "ئێوە": 2, "تۆش": 2, "تۆیش": 2, "ئێوەش": 2,
    "ئەو": 3, "ئەوان": 3, "ئەویش": 3, "ئەوانیش": 3,
}

# F#35 (Academy Committee, pp. 80-106): happening (ڕوودان) verbs keep ێ
# in ALL persons (دەسووتێم). ترسان is excluded — standard دەترسم has
# no ێ in modern usage.
_RUUDAN_PRESENT_STEMS_CHECK = ("سووت", "شک", "خنک", "پس", "ڕژ")

# F#36 (Academy Committee, pp. 96-107): suppletive causative pairs — the
# اندن template never applies to these bases.
_SUPPLETIVE_CAUSATIVES = (
    ("هاتاند", "هێنا"), ("چوواند", "برد"), ("ڕۆیشتاند", "نارد"),
    ("کەوتاند", "خست"), ("نووستاند", "نواند"),
)

# F#168 (Rasul, p. 25): خواردن keeps خۆ/خوا in the present — the ا→ێ
# alternation does not reach it (base form خوەردن).
_XWARDIN_ERRORS = {
    "دەخوێم": "دەخۆم", "دەخوێی": "دەخۆی", "دەخوێین": "دەخۆین",
    "دەخوێت": "دەخوات", "ئەخوێم": "ئەخۆم", "ئەخوێی": "ئەخۆی",
    "ناخوێم": "ناخۆم",
}

# R18/F#46 (Haji Marf 2014, pp. 116-121): in preverbed past transitives
# the Set 1 clitic sits between preverb and stem (هەڵمانگرت). Nominal
# lexicalisations that embed a transitive stem are excluded.
_PREVERBS = ("هەڵ", "دا", "ڕا", "دەر", "وەر", "تێ", "لێ", "پێ")
_PREVERB_NOMINAL_EXCLUSIONS = ("تێبینی", "دابین", "هەڵوێست", "پێویست",
                               "ڕاوێژ")

# F#76 (Slevanayi 2001, pp. 16, 72-73): plural vocatives demand a plural
# imperative. Closed list — ینە is also a feminine name ending (ژینە).
_PLURAL_VOCATIVES = {"کوڕینە", "کچینە", "هاوڕێینە", "خەڵکینە",
                     "براینە", "خوشکینە"}

# F#77 (Slevanayi 2001, pp. 87-88): numeral subjects force a plural verb.
# Time nouns are excluded (duration adverbials: دوو ڕۆژ مایەوە).
_NUMERALS_PLURAL = ("دوو", "سێ", "چوار", "پێنج", "شەش", "حەوت",
                    "هەشت", "نۆ", "دە")
_TIME_NOUNS = ("ڕۆژ", "شەو", "ساڵ", "مانگ", "هەفتە", "کاتژمێر",
               "خولەک", "چرکە", "جار", "ساعات", "دەقیقە")

# F#10-frame: stop categories for the demonstrative closing-ە rule
# (pronouns, quantifiers, numerals, titles, connectors).
_DEM_FRAME_STOP = {
    "من", "تۆ", "ئەو", "ئێمە", "ئێوە", "ئەوان", "خۆی", "خۆم", "خۆت",
    "یەک", "هەموو", "هەندێک", "چەند", "زۆر", "کەم", "هەر", "هیچ",
    "دوو", "سێ", "چوار", "پێنج", "شەش", "حەوت", "هەشت", "نۆ", "دە",
    "دکتۆر", "مامۆستا", "پرۆفیسۆر", "شێخ", "مەلا", "حاجی", "کاک",
    "خاتوو", "و", "کە", "یان", "بەڵام",
}
# F#123 (Haji Marf 2014, pp. 263-264): lexicalised dem+time compounds.
_DEM_FUSED_TIME = {"ساڵ": "ئەمساڵ", "شەو": "ئەمشەو", "ڕۆ": "ئەمڕۆ",
                   "جار": "ئەمجارە"}


@dataclass
class AgreementResult:
    """Result of agreement checking on a single sentence.

    ``checks_total`` is the number of checks run (always 15). ``checks_applicable``
    is how many of those checks had their trigger structure present in the
    sentence — the denominator that matters. A check that never fired (no
    pronoun, no quantifier, no relative clause, …) is neither a pass nor a
    fail; counting it as a silent pass is what inflated the old headline
    toward 1.0. ``accuracy`` divides passed checks by *applicable* checks.
    """
    sentence: str
    checks_passed: int
    checks_total: int
    violations: list[str]
    checks_applicable: int = 0

    @property
    def accuracy(self) -> float:
        # Denominator = applicable checks, not all 15. A sentence that
        # triggered no check is undefined for this metric; callers should
        # gate on is_applicable. To keep the property total we return 1.0
        # when nothing applied.
        if self.checks_applicable <= 0:
            return 1.0
        failed = self.checks_total - self.checks_passed
        passed_applicable = self.checks_applicable - failed
        return passed_applicable / self.checks_applicable

    @property
    def is_applicable(self) -> bool:
        return self.checks_applicable > 0

    @property
    def is_correct(self) -> bool:
        return len(self.violations) == 0


class AgreementChecker:
    """Check Central Kurdish (Sorani) agreement constraints in sentences.
    
    Uses rule-based morphological analysis to detect:
    1. Subject-verb person/number mismatch (Law 1 — Slevanayi 2001, p. 89)
    2. Clitic person consistency (F#9, F#133 — same-set exclusion)
    3. Ezafe presence/absence (F#165 — ی/یی six scenarios)
    4. Tense marker consistency within a clause (F#254 — coordination tense)
    
    Background agreement findings informing these checks:
      F#81  — Case determines agreement domain (nom ↔ external, obl ↔ internal)
      F#177 — Sorani lacks morphological case marking (cross-dialectal)
      F#205 — Inanimate nouns carry arbitrary grammatical gender
      F#206 — Post-head determiners always in nominative case internally
    """
    
    def __init__(self, analyzer: Optional[MorphologicalAnalyzer] = None):
        self._analyzer = analyzer or MorphologicalAnalyzer(use_klpt=False)
    
    def check_sentence(self, sentence: str) -> AgreementResult:
        """Run all agreement checks on a sentence.

        Each check returns ``(applicable, violations)``. ``applicable`` is
        True only when the check's trigger structure is present (a pronoun
        for subject-verb, a quantifier for quantifier-noun, …). Checks that
        do not apply are excluded from the denominator instead of silently
        counting as passes.
        """
        violations: list[str] = []
        failed_checks = 0
        applicable_checks = 0
        total_checks = 0

        checks = [
            self._check_subject_verb,
            self._check_clitic_consistency,
            self._check_ezafe,
            self._check_tense_consistency,
            self._check_object_verb_ergative,
            self._check_negative_concord,
            self._check_orthography,
            self._check_conditional_agreement,
            self._check_quantifier_noun,
            self._check_relative_clause,
            self._check_vocative_imperative,
            self._check_adverb_verb_tense,
            self._check_compound_subject,
            self._check_bare_noun_agreement,
            self._check_noun_subject_verb_number,
        ]

        for check in checks:
            applicable, check_violations = check(sentence)
            violations.extend(check_violations)
            total_checks += 1
            if applicable:
                applicable_checks += 1
            if check_violations:
                failed_checks += 1

        passed = total_checks - failed_checks

        return AgreementResult(
            sentence=sentence,
            checks_passed=passed,
            checks_total=total_checks,
            violations=violations,
            checks_applicable=applicable_checks,
        )
    
    def _clause_boundary_indices(self, words: list[str]) -> list[int]:
        """Return word indices that are clause boundaries.

        A clause boundary occurs at و when the preceding clause segment
        contains verb evidence OR when the segment is long enough to be
        a verbless clause (nominal/prepositional predicate). Punctuation
        (،/./?/!) also marks boundaries.
        """
        verb_prefixes = ("دە", "ئە", "نا", "بی", "بە", "ب")
        boundaries: list[int] = []
        segment_has_verb = False
        segment_word_count = 0
        for i, word in enumerate(words):
            if word in ("،", ".", "؟", "!"):
                boundaries.append(i)
                segment_has_verb = False
                segment_word_count = 0
            elif word == "و" and i > 0:
                if segment_has_verb or segment_word_count >= 2:
                    boundaries.append(i)
                    segment_has_verb = False
                    segment_word_count = 0
                # else: single-word NP-internal و (e.g. "نان و پەنیر")
            else:
                segment_word_count += 1
                if any(word.startswith(vp) for vp in verb_prefixes):
                    segment_has_verb = True
                for stem in TRANSITIVE_PAST_STEMS:
                    if word.startswith(stem):
                        segment_has_verb = True
                        break
        return boundaries

    def _check_subject_verb(self, sentence: str) -> tuple[bool, list[str]]:
        """Check subject-verb person/number agreement (Law 1).
        
        Source: Slevanayi (2001), p. 89 — verb agrees with subject in
        person and number for intransitive and present-tense transitive.

        Scans forward from each pronoun to the next clause boundary
        (instead of a fixed 6-word window) to avoid missing distant verbs.

        Applicable only when a subject pronoun is paired with a
        present-tense verb inside the same clause.
        """
        violations = []
        applicable = False
        words = self._analyzer.tokenize(sentence)
        clause_bounds = set(self._clause_boundary_indices(words))

        # F#128 (Haji Marf 2014, pp. 296-297): reciprocal یەکتر requires a
        # plural subject. A singular subject pronoun in the same clause as a
        # reciprocal (bare or clitic-hosting, e.g. یەکترم) is an agreement
        # error: *من یەکترم بینی.
        def _is_reciprocal_token(tok: str) -> bool:
            for base in RECIPROCAL_VARIANTS:
                if tok == base:
                    return True
                if tok.startswith(base) and tok[len(base):] in CLITIC_PERSON_MAP:
                    return True
            return False

        for i, word in enumerate(words):
            if word not in _PRONOUN_AGREEMENT:
                continue
            p_person, p_number = _PRONOUN_AGREEMENT[word]
            if p_number != "sg":
                continue
            for j in range(i + 1, len(words)):
                if j in clause_bounds:
                    break
                if _is_reciprocal_token(words[j]):
                    applicable = True
                    violations.append(
                        f"Reciprocal with singular subject: '{word}' "
                        f"with '{words[j]}' (F#128)"
                    )
                    break

        # F#50 (Haji Marf 2014, pp. 49-50): in verbless (copular) sentences
        # the Set-2 copular clitic on the predicate must match the subject:
        # من کوردم / تۆ کوردی(ت) / ئەو کوردە / ئێمە کوردین / ئێوە کوردن.
        # *من کوردە and *ئێمە کوردن are copular agreement errors.
        _COPULA_ENDINGS: tuple[tuple[str, frozenset], ...] = (
            ("ین", frozenset({("1", "pl")})),
            ("یت", frozenset({("2", "sg")})),
            ("م", frozenset({("1", "sg")})),
            ("ی", frozenset({("2", "sg")})),
            ("ن", frozenset({("2", "pl"), ("3", "pl")})),
            ("ە", frozenset({("3", "sg")})),
        )
        # F#72 (Slevanayi 2001, pp. 75-77): possessive هەبوون agrees with
        # the POSSESSED noun, never the possessor: کتێبەکانم هەن /
        # کتێبەکەم هەیە — *کتێبەکانم هەیە is a number clash.
        _HEBUN_NUMBER = {"هەیە": "sg", "نییە": "sg", "هەن": "pl", "نین": "pl"}
        for i, word in enumerate(words):
            num = _HEBUN_NUMBER.get(word)
            if num is None or i == 0:
                continue
            host = words[i - 1]
            base = host
            for cl in ("مان", "تان", "یان", "م", "ت", "ی"):
                if host.endswith(cl) and len(host) > len(cl) + 3:
                    base = host[: -len(cl)]
                    break
            if base.endswith("ەکان"):
                host_num = "pl"
            elif base.endswith("ەکە"):
                host_num = "sg"
            else:
                continue
            applicable = True
            if host_num != num:
                violations.append(
                    f"Possessed-noun agreement: '{host}' ({host_num}) with "
                    f"'{word}' — هەبوون agrees with the possessed noun "
                    f"(F#72)"
                )

        # F#39 tables / F#27 (Academy Committee, pp. 144-156; Rasul 2004):
        # past intransitives take a Set 2 suffix matching the subject:
        # من هاتم / تۆ هاتیت / ئەو هات — *من هات and *ئەو ڕۆیشتی clash.
        # Only the immediately following word is checked, and only in
        # subject-licensing contexts (بۆ من نارد, براکەی من هات skipped).
        for i, word in enumerate(words):
            if word not in _PRONOUN_AGREEMENT or i + 1 >= len(words):
                continue
            if i > 0 and words[i - 1] not in _SUBJ_CONTEXT_OK:
                continue
            nxt = words[i + 1]
            parsed = None
            for stem in _PAST_INTRANS_STEMS_CHECK:
                if nxt.startswith(stem):
                    sfx = nxt[len(stem):]
                    if sfx in _PAST_SET2_SUFFIXES:
                        parsed = (stem, sfx)
                    break
            if parsed is None:
                continue
            stem, sfx = parsed
            expected = _PRONOUN_AGREEMENT[word]
            applicable = True
            if expected not in _PAST_SET2_SUFFIXES[sfx]:
                good = f"{stem}{_PRONOUN_PAST_SUFFIX[expected]}"
                violations.append(
                    f"Past subject-verb mismatch: '{word}' with '{nxt}' — "
                    f"the Set 2 suffix must match the subject ('{good}') "
                    f"(F#39/F#27)"
                )

        for i, word in enumerate(words):
            # ئەو doubles as a demonstrative determiner — too ambiguous.
            if word not in _PRONOUN_AGREEMENT or word == "ئەو":
                continue
            if i + 1 >= len(words):
                continue
            pred = words[i + 1]
            # Copular reading requires the predicate to be clause-final.
            if i + 2 < len(words) and words[i + 2] not in {"و", "،", ".", "؟", "!"}:
                continue
            pf = self._analyzer.analyze_token(pred)
            if pf.pos == "VERB" or _is_present_verb(pred) or _is_past_verb(pred, pf):
                continue
            # Set-1 possessive hosts (ماڵمان) are not copular predicates.
            if pred.endswith(("مان", "تان", "یان")):
                continue
            expected = (_PRONOUN_AGREEMENT[word][0], _PRONOUN_AGREEMENT[word][1])
            for ending, pns in _COPULA_ENDINGS:
                if pred.endswith(ending) and len(pred) > len(ending) + 1:
                    applicable = True
                    if expected not in pns:
                        violations.append(
                            f"Copular clitic mismatch: '{word}' with "
                            f"predicate '{pred}' (ـ{ending}) (F#50)"
                        )
                    break

        for i, word in enumerate(words):
            if word not in _PRONOUN_AGREEMENT:
                continue
            expected_person, expected_number = _PRONOUN_AGREEMENT[word]
            
            # Scan forward for a verb until clause boundary
            for j in range(i + 1, len(words)):
                if j in clause_bounds:
                    break
                candidate = words[j]
                is_present = any(candidate.startswith(p) for p in _PRESENT_PREFIXES)
                is_neg_present = candidate.startswith(_NEGATION_PRESENT_PREFIX)
                
                if not (is_present or is_neg_present):
                    continue
                
                # Extract verb ending → person/number
                verb_pn = self._verb_ending_to_pn(candidate)
                if verb_pn is None:
                    break
                
                applicable = True
                verb_person, verb_number = verb_pn
                if verb_person != expected_person or verb_number != expected_number:
                    violations.append(
                        f"Subject-verb mismatch: '{word}' ({expected_person}{expected_number}) "
                        f"with verb '{candidate}' ({verb_person}{verb_number})"
                    )
                break
        
        return applicable, violations
    
    @staticmethod
    def _verb_ending_to_pn(verb: str) -> Optional[tuple[str, str]]:
        """Extract person/number from a verb's ending suffix."""
        # Check longest suffixes first
        for suffix in sorted(_PRESENT_ENDINGS, key=len, reverse=True):
            if verb.endswith(suffix):
                return _PRESENT_ENDINGS[suffix]
        return None
    
    def _check_clitic_consistency(self, sentence: str) -> tuple[bool, list[str]]:
        """Check for inconsistent clitic usage within a clause.

        Rules enforced:
        - F#133: Same-set clitic exclusion — two clitics from the same set
          (e.g., two Set 1 clitics) cannot co-occur in a simple sentence.
        - F#9: Clitic person must be plausible in context (no two different
          person clitics on adjacent words unless compounding).

        EVAL-5 fixes:
        - Past verbs also carry Set 2 suffixes (not Set 1 clitics); skip them.
        - Uses analyzer's morphological features for ی disambiguation instead
          of bare endswith("ی"), avoiding false filtering on ezafe/indefinite ی.
        """
        violations = []
        applicable = False
        words = self._analyzer.tokenize(sentence)
        found_clitics: list[tuple[str, str, str]] = []  # (clitic, person, number)

        # F#124 (Haji Marf 2014, pp. 291-293): the possessive pronoun هی/ئی
        # can NEVER host a clitic — *هیم, *هیتان are ungrammatical. The
        # correct form is هی + independent pronoun (هی من). The same ban
        # covers ش/یش (*هیش).
        for word in words:
            if word in CLITIC_BARRED_PRONOUNS:
                applicable = True     # correct usage present — check applied
                continue
            if word in ("هیش", "هییش"):
                applicable = True
                violations.append(
                    f"Clitic-barred pronoun: 'هی' takes neither clitics "
                    f"nor ش/یش ('{word}') (F#124)"
                )
                continue
            for base in CLITIC_BARRED_PRONOUNS:
                if not word.startswith(base) or word == base:
                    continue
                rest = word[len(base):]
                # Allow a final copular ە after the clitic (هیمە "it's mine")
                if rest.endswith("ە") and rest[:-1] in CLITIC_PERSON_MAP:
                    rest = rest[:-1]
                if rest in CLITIC_PERSON_MAP:
                    applicable = True
                    violations.append(
                        f"Clitic-barred pronoun: '{base}' cannot host "
                        f"clitic '{rest}' (F#124)"
                    )
                    break

        # F#52 (Haji Marf 2014, pp. 164-167): Set 2 clitics never stack —
        # *دەچمین doubles 1sg م and 1pl ین on one present verb.
        for word in words:
            for pre in ("دە", "ئە", "نا"):
                if not word.startswith(pre):
                    continue
                rest = word[len(pre):]
                for stem in _PRESENT_STEMS_NON_B:
                    if rest.startswith(stem) and rest[len(stem):] == "مین":
                        applicable = True
                        violations.append(
                            f"Double Set 2 clitic: '{word}' — م and ین "
                            f"cannot stack ('{pre}{stem}م' / "
                            f"'{pre}{stem}ین') (F#52)"
                        )
                        break
                break

        # F#39 (Academy Committee, pp. 144-156): under negation the Set 1
        # agent clitic PRECEDES the transitive past stem: نەمگرت, نەیانکرد —
        # *نەگرتم/*نەکردیان leave the agent slot empty. Perfects (نەگرتوومە)
        # and subjunctives (نەگرتبام) are skipped because the remainder
        # after the stem must be exactly a clitic.
        for word in words:
            if not word.startswith("نە") or len(word) < 6:
                continue
            rest = word[2:]
            for stem in TRANSITIVE_PAST_STEMS:
                if len(stem) < 3 or not rest.startswith(stem):
                    continue
                suffix = rest[len(stem):]
                if suffix in ("م", "ی", "ت", "مان", "تان", "یان"):
                    applicable = True
                    violations.append(
                        f"Negated transitive clitic: '{word}' — under "
                        f"negation the agent clitic precedes the stem "
                        f"('نە{suffix}{stem}') (F#39)"
                    )
                break

        # R18/F#46 (Haji Marf 2014, pp. 116-121): in preverbed past
        # transitives the Set 1 clitic sits between preverb and stem:
        # هەڵمانگرت — *هەڵگرتمان strands it. Only the unambiguous
        # plural clitics are matched; nominal lexicalisations (تێبینی,
        # پێویست) are excluded.
        for word in words:
            if any(word.startswith(x) for x in _PREVERB_NOMINAL_EXCLUSIONS):
                continue
            for pv in _PREVERBS:
                if not word.startswith(pv) or len(word) < len(pv) + 5:
                    continue
                rest = word[len(pv):]
                for stem in TRANSITIVE_PAST_STEMS:
                    if len(stem) < 3 or not rest.startswith(stem):
                        continue
                    sfx = rest[len(stem):]
                    if sfx in ("مان", "تان", "یان"):
                        applicable = True
                        violations.append(
                            f"Preverb clitic position: '{word}' — the "
                            f"Set 1 clitic sits between preverb and stem "
                            f"('{pv}{sfx}{stem}') (R18/F#46)"
                        )
                    break
                break

        for word in words:
            # Set 2 verb suffixes (م/ت/ی on دەکەم etc.) are NOT
            # Set 1 clitics. Skip present-tense verbs entirely.
            if _is_present_verb(word) or word.startswith(_NEGATION_PRESENT_PREFIX):
                continue
            # Also skip past verbs — their suffixes are Set 2 agreement,
            # not Set 1 clitics (EVAL-5 fix: prevents Set 2 leak).
            if _is_transitive_past(word):
                continue
            # Any word the analyzer reads as a verb carries agreement suffixes
            # (Set 2 subject on چووم, or the ergative slot), never a mobile
            # Set 1 clitic. Skipping verbs stops the false F#133 flag on
            # correct sentences like «چووم … خزمانم کرد».
            wf = self._analyzer.analyze_token(word)
            if wf.pos == "VERB":
                continue
            for cl, (person, number) in CLITIC_PERSON_MAP.items():
                if word.endswith(cl) and len(word) > len(cl) + 1:
                    # Use analyzer's morphological features to detect
                    # possessive / ezafe (Set 3) constructions. The old bare
                    # endswith("ی") check was ambiguous (EVAL-5 fix).
                    if cl == "ی":
                        # ezafe / ambiguous ی (سەردانی خزمانم) is a linker,
                        # not a Set 1 clitic.
                        if wf.raw_analysis.get("yi_ambiguous") or wf.case == "ez":
                            continue
                    else:
                        stem = word[: -len(cl)]
                        feats = self._analyzer.analyze_token(stem + "ی")
                        if feats.case == "ez":
                            # Stem has ezafe case → possessive, not Set 1
                            continue
                    found_clitics.append((cl, person, number))
                    break

        # Same-set exclusion check (F#133): two Set 1 clitics cannot
        # co-occur in a simple sentence. Flag if we see two or more
        # distinct clitics — either with different persons or the same
        # clitic appearing on multiple hosts.
        if len(found_clitics) >= 2:
            applicable = True
            persons_seen = {c[1] for c in found_clitics}
            distinct_clitics = {c[0] for c in found_clitics}
            if len(persons_seen) >= 2 or len(distinct_clitics) >= 2 or len(found_clitics) > len(distinct_clitics):
                violations.append(
                    f"Clitic inconsistency: {len(found_clitics)} Set 1 clitics "
                    f"with {len(distinct_clitics)} distinct forms and "
                    f"{len(persons_seen)} person(s) in one clause (F#133)"
                )

        # F#127 (Haji Marf 2014, pp. 215-216): with the -ەوە suffix the Set-1
        # clitic goes BEFORE the suffix — کردمانەوە, never *کردەوەمان.
        for word in words:
            for cl in CLITIC_PERSON_MAP:
                if not word.endswith(cl) or len(word) <= len(cl) + 3:
                    continue
                rest = word[: -len(cl)]
                if not rest.endswith("ەوە"):
                    continue
                # Set-1 clitics are past-transitive agents, so the stem before
                # ەوە must be a transitive past stem (کرد, گرت…). This also
                # rules out noun hosts like ماڵەوەم (ماڵ is not a past stem).
                stem = rest[:-3]
                if _is_transitive_past(stem):
                    applicable = True
                    violations.append(
                        f"Clitic position: '{word}' — clitic must precede ەوە "
                        f"({stem}{cl}ەوە) (F#127)"
                    )
                break

        # F#122 (Haji Marf 2014, pp. 245-263): on demonstratives ش/یش must
        # precede the clitic — ئەمەشم ✓, *ئەمەمیش ✗. (خۆ alone allows both
        # orders — خۆشم/خۆمیش — and is not a demonstrative, so unaffected.)
        _DEM_BASES = ("ئەمانە", "ئەوانە", "ئەمە", "ئەوە", "ئەم", "ئەو")
        for word in words:
            for base in _DEM_BASES:
                if not word.startswith(base) or word == base:
                    continue
                rest = word[len(base):]
                for cl in CLITIC_PERSON_MAP:
                    if rest == cl + "یش" or rest == cl + "ش":
                        applicable = True
                        violations.append(
                            f"ش/یش order: '{word}' — ش/یش must precede the "
                            f"clitic on demonstratives ({base}ش{cl}) (F#122)"
                        )
                        break
                break

        # Cross-clause covert-subject consistency (F#22, Amin 2016; Rasul
        # 2005 pp. 13-14): و-coordinated clauses sharing a DROPPED subject
        # must mark it consistently — the Set-2 subject suffix of an
        # intransitive/present clause (چووین → 1pl) and the Set-1 agent
        # clitic of a past-transitive clause (سەردانی خزمانم کرد → م 1sg)
        # must agree in person and number. *چووین … خزمانم کرد mixes ئێمە
        # with من; correct: خزمانمان کرد or چووم.
        bounds = sorted(self._clause_boundary_indices(words))
        if bounds:
            try:
                graph = build_agreement_graph(sentence, self._analyzer)
            except (KeyError, AttributeError, TypeError, IndexError):
                graph = None
            if graph is not None and len(graph.tokens) == len(words):
                def _clause_of(idx: int) -> int:
                    return sum(1 for b in bounds if idx > b)

                feats = graph.features
                overt_subj_verbs = {
                    e.target_idx for e in graph.edges
                    if e.agreement_type in (
                        "subject_verb", "passive_subject_verb",
                        "backward_subject_verb", "agent_non_agreeing",
                    )
                }
                signatures: list[tuple[str, str, str]] = []  # (person, number, marker)
                for vi, f in enumerate(feats):
                    if f.pos != "VERB" or vi in overt_subj_verbs:
                        continue
                    if getattr(f, "voice", "") == "passive":
                        continue          # passive demotes the agent
                    # Transitivity: analyzer/lexicon first, heuristic stem
                    # list as fallback (lexicon-less analyzers leave it '').
                    is_past_trans = f.tense == "past" and (
                        f.transitivity == "trans"
                        or (not f.transitivity and _is_transitive_past(words[vi]))
                    )
                    if is_past_trans:
                        # Dropped agent: its Set-1 clitic lodges on a host
                        # earlier in the same clause (F#126/F#129).
                        for j in range(vi - 1, -1, -1):
                            if _clause_of(j) != _clause_of(vi):
                                break
                            fj = feats[j]
                            hosted = (getattr(fj, "raw_analysis", None) or {}).get("hosted_clitic", "")
                            if getattr(fj, "is_clitic", False) and hosted in CLITIC_PERSON_MAP:
                                p, nnum = CLITIC_PERSON_MAP[hosted]
                                signatures.append((p, nnum, f"{hosted} ({words[vi]})"))
                                break
                    elif f.person and f.number:
                        signatures.append((f.person, f.number, words[vi]))
                if len(signatures) >= 2:
                    applicable = True
                    if len({(p, nnum) for p, nnum, _ in signatures}) > 1:
                        a, b = signatures[0], signatures[1]
                        violations.append(
                            f"Cross-clause covert-subject mismatch: '{a[2]}' "
                            f"({a[0]}{a[1]}) with '{b[2]}' ({b[0]}{b[1]}) (F#22)"
                        )

        return applicable, violations
    
    def _check_ezafe(self, sentence: str) -> tuple[bool, list[str]]:
        """Check for ezafe (ی/یی) issues in noun phrases.
        
        Rules enforced:
        - F#165: After consonant-final noun, ezafe is ی; after vowel-final, یی.
        - F#10/R4: Demonstrative (ئەم/ئەو) cannot co-occur with ەکە/ێک.
        - Missing ezafe between noun and attributive adjective is an error.

        Applicable when a demonstrative NP is present or a word carries an
        ezafe-ی before a modifier.
        """
        violations = []
        applicable = False
        words = self._analyzer.tokenize(sentence)
        
        # Check demonstrative + definiteness co-occurrence (F#10, Rule R4)
        demonstratives = {"ئەم", "ئەو", "ئەمە", "ئەوە"}
        definite_markers = ("ەکە", "یەکە", "ەکان", "یەکان")
        # ێکە covers the frame-final form *ئەم کوڕێکە (ێک + closing ە);
        # the numeral یەکێک(ە) "one (of)" is a legitimate copular host.
        indefinite_markers = ("ێک", "یەک", "ێکی", "ێکە")
        _INDEF_EXEMPT = {"یەکێک", "یەکێکە", "یەکێکیان"}
        
        in_dem_np = False
        dem_word = ""
        dem_words_seen = 0
        for i, word in enumerate(words):
            if word in demonstratives:
                in_dem_np = True
                applicable = True
                dem_word = word
                dem_words_seen = 0
                continue
            
            if in_dem_np:
                dem_words_seen += 1
                # Within a demonstrative NP, check for prohibited markers
                has_def = any(word.endswith(m) for m in definite_markers)
                has_indef = (
                    any(word.endswith(m) for m in indefinite_markers)
                    and word not in _INDEF_EXEMPT
                )
                if has_def:
                    violations.append(
                        f"Demonstrative+definite co-occurrence: '{dem_word}' with "
                        f"definite marker on '{word}' (F#10/R4)"
                    )
                if has_indef:
                    violations.append(
                        f"Demonstrative+indefinite co-occurrence: '{dem_word}' with "
                        f"indefinite marker on '{word}' (F#10/R4)"
                    )
                # Demonstrative NPs in Sorani can span up to ~3 words
                # (dem + adj* + noun + closing -ə). End tracking after
                # 3 non-demonstrative words or at a clause boundary.
                if dem_words_seen >= 3 or word in {"و", "،", ".", "؟"}:
                    in_dem_np = False
        
        # Check ezafe allomorph: ی after consonant, یی after vowel (F#165)
        # Kurdish vowel characters at word-final position
        _vowels = {"ا", "ە", "ێ", "ی", "ۆ", "و"}
        for i, word in enumerate(words):
            if i + 1 >= len(words):
                continue
            # Check if word ends with ezafe ی and is followed by a modifier
            if not word.endswith("ی") or len(word) < 2:
                continue
            base = word[:-1]
            if not base:
                continue
            # F#49/R20 (Haji Marf 2014, p. 107): pronouns take single ی-ezafe
            # regardless of their final vowel (تۆی باش is correct, never *تۆیی).
            # The یی allomorph rule applies to nouns only.
            if base in SUBJECT_PRONOUNS:
                continue
            # Only check when next word looks like a modifier (not a verb)
            next_word = words[i + 1]
            if next_word.startswith(("دە", "بی", "نا", "نە", "مە")):
                continue
            # Ezafe links a noun to a following modifier, never to a verb:
            # in «نامەکەی نووسی» the ی is the 3sg possessive/agent clitic,
            # not ezafe, so the allomorph rule does not apply.
            nf = self._analyzer.analyze_token(next_word)
            if nf.pos == "VERB" or _is_past_verb(next_word, nf):
                continue
            final_char = base[-1]
            if final_char in _vowels and not word.endswith("یی"):
                applicable = True
                violations.append(
                    f"Ezafe allomorph: vowel-final '{base}' should take یی "
                    f"not single ی (F#165)"
                )

        # F#49/R20 (Haji Marf 2014, p. 107): pronouns NEVER take ە-ezafe.
        # *تۆە باش is always wrong; the correct linker is ی (تۆی باش).
        # Only flagged before a modifier — a clause-final pronoun+ە is the
        # 3sg copula (ئەوە تۆە "that's you"), which is grammatical (F#50).
        for i, word in enumerate(words):
            if i + 1 >= len(words) or len(word) < 2 or not word.endswith("ە"):
                continue
            base = word[:-1]
            if base not in SUBJECT_PRONOUNS or word in SUBJECT_PRONOUNS:
                continue
            next_word = words[i + 1]
            if next_word in {"و", "،", ".", "؟", "!"}:
                continue
            if next_word.startswith(("دە", "بی", "نا", "نە", "مە")):
                continue
            applicable = True
            violations.append(
                f"Pronoun ezafe: '{base}' takes ی-ezafe, never ە (F#49)"
            )

        # F#60 (Mamajalakayi; Ibrahim 1988): definite/indefinite allomorph
        # selection — ەکە/ێک after consonant-final stems, یەکە/یەک only after
        # vowel-final stems. *کتێبیەکە and *کتێبیەک are misattachments.
        for word in words:
            for suf, correct in (("یەکە", "ەکە"), ("یەک", "ێک")):
                if not word.endswith(suf) or word == suf:
                    continue
                stem = word[: -len(suf)]
                if len(stem) < 2:
                    continue
                if stem[-1] in _vowels:
                    continue          # vowel-final → ی-allomorph is correct
                applicable = True
                violations.append(
                    f"Determiner allomorph: consonant-final '{stem}' takes "
                    f"{correct} not {suf} (F#60)"
                )
                break

        # F#86 (Slevanayi 2001, pp. 43-44): proper nouns denote a unique
        # entity and can NEVER take indefinite or plural markers:
        # *هەولێرێک, *دهۆکان, *هەولێرەکان. Gentilic forms carry a double
        # یی (هەولێرییەکان "the Erbilites") and are not matched here.
        for word in words:
            for name in _PROPER_PLACE_NAMES:
                if not word.startswith(name) or word == name:
                    continue
                rest = word[len(name):]
                if name[-1] in _vowels:
                    banned = ("یەک", "یەکان")
                else:
                    banned = ("ێک", "ەکان", "ان")
                if rest in banned:
                    applicable = True
                    violations.append(
                        f"Proper noun morphology: '{word}' — the proper "
                        f"noun '{name}' cannot take indefinite/plural "
                        f"markers (F#86)"
                    )
                break

        # F#123 (Haji Marf 2014, pp. 263-264): ئەم + time noun is a
        # lexicalised compound — writing ئەمساڵ/ئەمشەو/ئەمڕۆ as two
        # words is a segmentation error.
        for i, word in enumerate(words):
            if word != "ئەم" or i + 1 >= len(words):
                continue
            fused = _DEM_FUSED_TIME.get(words[i + 1])
            if fused is None:
                continue
            if i + 2 < len(words) and words[i + 2] == "و":
                continue
            applicable = True
            violations.append(
                f"Fused time word: 'ئەم {words[i + 1]}' — lexicalised as "
                f"one word ('{fused}') (F#123)"
            )

        # F#10-frame (Amin 1986, pp. 24-25): the demonstrative frame must
        # close with ە: لەم شارە دەژیم — *لەم شار دەژیم strands it. Bare
        # ئەم/ئەو additionally require a non-verbal continuation so
        # pronoun readings with bare objects (ئەو نان دەخوات) stay clean.
        _verb_like = ("دە", "ئە", "نا", "نە", "مە")
        for i, word in enumerate(words):
            if word not in ("لەم", "بەم", "لەو", "بەو", "ئەم", "ئەو"):
                continue
            if i + 1 >= len(words):
                continue
            w = words[i + 1]
            if w in _DEM_FRAME_STOP or len(w) < 3:
                continue
            if "ە" in w[-3:] or w.endswith(("ی", "ێ")):
                continue
            if w.startswith(_verb_like):
                continue
            if word in ("ئەم", "ئەو") and w.startswith("ب"):
                continue
            # The analyzer misparses some nouns (شوێن) as VERB — rely on
            # the past-verb detector plus the prefix guards above instead.
            if _is_past_verb(w, self._analyzer.analyze_token(w)):
                continue
            if i + 2 < len(words):
                after = words[i + 2]
                if after == "و":
                    continue
                if word in ("ئەم", "ئەو") and (
                        after.startswith(_verb_like)
                        or _is_past_verb(
                            after, self._analyzer.analyze_token(after))):
                    continue
            elif word in ("ئەم", "ئەو"):
                continue    # bare dem + clause-final noun: too ambiguous
            applicable = True
            linker = "یە" if w[-1] in ("ا", "ۆ", "و") else "ە"
            violations.append(
                f"Demonstrative frame: '{word} {w}' — the frame closes "
                f"with ە ('{w}{linker}') (F#10)"
            )

        # F#90 (Slevanayi 2001, pp. 47-48): definiteness never doubles —
        # no word stacks two ەکان sequences.
        for word in words:
            if "ەکانەکان" in word:
                applicable = True
                violations.append(
                    f"Double plural-definite: '{word}' — ەکان cannot "
                    f"stack (F#90)"
                )

        return applicable, violations
    
    def _check_tense_consistency(self, sentence: str) -> tuple[bool, list[str]]:
        """Check tense marker consistency within a clause.
        
        Rules enforced:
        - F#254: In و-coordinated clauses, non-past → past sequence is
          ungrammatical (Maaruf 2009, pp. 84-85). Only past→non-past and
          same-tense are valid.
        - Mixed tense prefixes (دە/ئە with past stem, or no prefix with
          present stem) within a single clause flag inconsistency.
        """
        violations = []
        applicable = False
        words = self._analyzer.tokenize(sentence)
        
        # Split on و to find coordinated clauses, but only when و
        # appears between verb-bearing segments (not NP-internal و).
        # Improved: only split when the preceding segment contains a verb,
        # preventing false splits on NP-internal "و" (e.g., "نان و پەنیر").
        verb_prefixes = ("دە", "ئە", "نا", "بی", "بە", "ب")
        clauses: list[list[str]] = [[]]
        for word in words:
            if word == "و" and len(clauses[-1]) > 1:
                # Only split if current clause has verb evidence
                has_verb = any(
                    w.startswith(vp) for w in clauses[-1] for vp in verb_prefixes
                )
                if has_verb:
                    clauses.append([])
                else:
                    clauses[-1].append(word)
            else:
                clauses[-1].append(word)
        
        # Filter out clauses that contain no verb evidence (likely NP
        # fragments from NP-internal و splits).
        clauses = [
            c for c in clauses
            if any(w.startswith(vp) for w in c for vp in verb_prefixes)
        ] or clauses  # keep original if no verb clauses found
        
        # Determine tense of each clause
        clause_tenses: list[Optional[str]] = []
        for clause_words in clauses:
            tense = self._detect_clause_tense(clause_words)
            clause_tenses.append(tense)
        
        # F#254: Check sequential tense ordering
        for i in range(len(clause_tenses) - 1):
            t1 = clause_tenses[i]
            t2 = clause_tenses[i + 1]
            if t1 and t2:
                applicable = True
                # Non-past followed by past is blocked
                if t1 == "present" and t2 == "past":
                    violations.append(
                        f"Tense sequencing violation: non-past clause followed by "
                        f"past clause in و-coordination (F#254)"
                    )

        # R17/F#40 (Academy Committee, pp. 151-153): the transitive present
        # perfect requires a final ە after the Set 1 clitic: گرتوومە,
        # کردوویانە — *گرتووم and *کردووی are missing it. Intransitive
        # perfects (هاتووم, کەوتووی) use Set 2 without ە and stay clean.
        for word in words:
            for cl in ("مان", "تان", "یان", "م", "ت", "ی"):
                if not word.endswith(cl):
                    continue
                rem = word[: -len(cl)]
                if (len(rem) >= 4 and rem.endswith("وو")
                        and _is_transitive_past(rem[:-2])):
                    applicable = True
                    violations.append(
                        f"Perfect missing copula: '{word}' — the transitive "
                        f"perfect requires final ە ('{word}ە') (R17/F#40)"
                    )
                break

        # F#35 (Academy Committee, pp. 80-106): happening (ڕوودان) verbs
        # keep ێ in ALL persons: دەسووتێم/دەسووتێین — *دەسووتم drops it.
        for word in words:
            for pre in ("دە", "ئە", "نا", "ب"):
                if not word.startswith(pre):
                    continue
                rest = word[len(pre):]
                for stem in _RUUDAN_PRESENT_STEMS_CHECK:
                    if rest.startswith(stem) and rest[len(stem):] in (
                            "م", "ی", "ین", "ن"):
                        applicable = True
                        sfx = rest[len(stem):]
                        violations.append(
                            f"Happening-verb conjugation: '{word}' — ڕوودان "
                            f"verbs take ێ in all persons "
                            f"('{pre}{stem}ێ{sfx}') (F#35)"
                        )
                        break
                break

        return applicable, violations
    
    @staticmethod
    def _detect_clause_tense(words: list[str]) -> Optional[str]:
        """Detect the dominant tense of a clause from verb morphology.

        Delegates to builder helpers for present-tense detection; falls back
        to stem matching for past tense.
        """
        for word in words:
            if _is_present_verb(word):
                return "present"
            # نا + verb stem = negated present (e.g. ناکات, نازانم).
            # Guard against false positives on nouns like نانی by requiring
            # a present-tense verb ending after the prefix.
            if word.startswith(_NEGATION_PRESENT_PREFIX) and len(word) > 3:
                remainder = word[len(_NEGATION_PRESENT_PREFIX):]
                if any(remainder.endswith(s) for s in _PRESENT_ENDINGS):
                    return "present"
            if word.startswith(_IMPERATIVE_PREFIX) and len(word) > 1:
                return "present"
        # Past: check for known transitive/intransitive past stems
        for word in words:
            if _is_present_verb(word):
                continue
            for stem in TRANSITIVE_PAST_STEMS:
                if word.startswith(stem) or (
                    word.startswith(_NEGATION_PAST_PREFIX)
                    and word[len(_NEGATION_PAST_PREFIX):].startswith(stem)
                ):
                    return "past"
            if any(word.endswith(p) for p in ("مان", "تان", "یان")) and len(word) > 4:
                return "past"
        return None

    def _check_object_verb_ergative(self, sentence: str) -> tuple[bool, list[str]]:
        """Check agent-clitic agreement in past transitive (ergative) clauses.

        Slevanayi (2001, pp. 60-61, 89) describes the split: in a past
        transitive clause the verb's *object* agreement is its inflectional
        ending (3sg = zero on a bare stem), while the agent is marked by a
        Set 1 enclitic. On a bare verb such as ``نووسیم`` the only overt
        person marker is that agent clitic ``م`` — not object agreement.

        So this check verifies the *agent* relation: when an overt subject
        pronoun (the agent) sits in the same clause as a past transitive
        verb carrying a Set 1 clitic, their person/number must match.
        Comparing the clitic against the object would conflate the agent
        marker with object agreement and falsely flag grammatical clauses
        (e.g. 1sg agent + 3sg object ``نامەکەم نووسی``); the check therefore
        scans backward to the clause boundary for the subject pronoun.

        The verb's object-agreement ending is not separable from the bare
        stem by surface heuristics, so object agreement itself is not scored
        here; it is the agent clitic that the generator perturbs and that
        this check validates.
        """
        violations = []
        applicable = False
        words = self._analyzer.tokenize(sentence)
        clause_bounds = set(self._clause_boundary_indices(words))

        for i, word in enumerate(words):
            # Skip present-tense verbs
            if _is_present_verb(word):
                continue
            if not _is_transitive_past(word):
                continue

            # The overt enclitic on a bare ergative verb is the Set 1 agent
            # clitic; extract its person/number.
            verb_pn = self._verb_ending_to_pn(word)
            if verb_pn is None:
                continue
            agent_person, agent_number = verb_pn

            # Scan backward to clause boundary for the overt subject (agent).
            for j in range(i - 1, -1, -1):
                if j in clause_bounds:
                    break
                subj = words[j]
                if subj not in _PRONOUN_AGREEMENT:
                    continue
                subj_person, subj_number = _PRONOUN_AGREEMENT[subj]
                applicable = True
                if subj_person != agent_person or subj_number != agent_number:
                    violations.append(
                        f"Ergative agent mismatch (Law 2 clause): subject "
                        f"'{subj}' ({subj_person}{subj_number}) with Set 1 "
                        f"agent clitic on '{word}' "
                        f"({agent_person}{agent_number})"
                    )
                break  # only the nearest overt subject is the agent

        return applicable, violations

    # ── PIPE-9: Additional checks for uncovered error generators ──

    # Common orthographic confusion pairs (subset of what orthography.py generates)
    _ORTHO_CONFUSIONS = [
        ("ح", "ه"),
        ("خ", "غ"),
        ("ڵ", "ل"),
        ("ڕ", "ر"),
        ("ع", "ئ"),
    ]

    def _check_orthography(self, sentence: str) -> tuple[bool, list[str]]:
        """Flag words containing likely orthographic confusions.

        Uses the lexicon (when available) to check whether a word with a
        known confusion character is misspelled. This catches cases where
        the model preserved a misspelling instead of correcting it.

        Applicable when a confusion character appears in some word.
        """
        violations = []
        applicable = False
        words = self._analyzer.tokenize(sentence)
        for word in words:
            for a, b in self._ORTHO_CONFUSIONS:
                if a in word:
                    applicable = True
                    alt = word.replace(a, b, 1)
                    if (self._analyzer._lexicon
                            and hasattr(self._analyzer._lexicon, 'is_correct')
                            and not self._analyzer._lexicon.is_correct(word)
                            and self._analyzer._lexicon.is_correct(alt)):
                        violations.append(
                            f"Orthographic confusion: '{word}' may be '{alt}' "
                            f"({a}→{b})"
                        )
                        break

        # F#155 (Rasul 2005, p. 21): the perfect-participle morpheme is و in
        # standard Sorani. The Northern ی and Southern گ allomorphs on a past
        # stem (*هاتیە, *مردگە) should be the و-form (هاتووە/مردووە).
        for word in words:
            for dial in ("یە", "گە"):
                if not word.endswith(dial) or len(word) <= len(dial) + 2:
                    continue
                stem = word[: -len(dial)]
                sf = self._analyzer.analyze_token(stem)
                # The remainder must be a bona-fide past stem (هات, مرد, ژیا)
                # — nouns like کوردیە/بەڵگە never match.
                if _is_past_verb(stem, sf):
                    applicable = True
                    violations.append(
                        f"Dialectal participle: '{word}' uses the "
                        f"{dial[0]}-allomorph; standard Sorani is "
                        f"'{stem}ووە' (F#155)"
                    )
                break

        # F#123 (Haji Marf 2014, pp. 263-264): بە/لە contract with a
        # following demonstrative — the initial ئە drops. Writing *بە ئەم
        # as two words instead of بەم is a segmentation error.
        for i in range(len(words) - 1):
            if words[i] in ("بە", "لە") and words[i + 1] in _DEM_FULL_FORMS:
                applicable = True
                contracted = words[i] + words[i + 1][2:]
                violations.append(
                    f"Demonstrative contraction: '{words[i]} {words[i + 1]}' "
                    f"should be '{contracted}' (F#123)"
                )

        # R19 (Academy Committee): ش before ت in the infinitive becomes ژ
        # in the present stem — دەکوژم/دەهاوێژم, never *دەکوشم. A ت right
        # after the stem is the past morpheme (دەیکوشت) and stays clean.
        for word in words:
            for pre in ("دە", "ئە", "نا", "نە", "مە", "ب"):
                if not word.startswith(pre):
                    continue
                rest = word[len(pre):]
                for stem in _SH_ZH_PAST_STEM_PREFIXES:
                    if not rest.startswith(stem):
                        continue
                    after = rest[len(stem):]
                    if after and not after.startswith("ت"):
                        applicable = True
                        fixed = pre + stem[:-1] + "ژ" + after
                        violations.append(
                            f"Present stem ش→ژ: '{word}' — the present "
                            f"stem uses ژ ('{fixed}') (R19)"
                        )
                    break
                break

        # R12 (Academy Committee): ە-final and ۆ-final present stems take
        # the ات 3sg allomorph (دەکات, دەخوات) — the raw sequences ەێ/ۆێ
        # (*دەکەێت, *دەخۆێت) are orthographically illegal in Sorani.
        for word in words:
            if "ەێ" in word:
                applicable = True
                violations.append(
                    f"3sg allomorph: '{word}' — ە-final stems take ات "
                    f"('{word.replace('ەێ', 'ا', 1)}') (R12)"
                )
            elif "ۆێ" in word:
                applicable = True
                violations.append(
                    f"3sg allomorph: '{word}' — ۆ-final stems take وات "
                    f"('{word.replace('ۆێ', 'وا', 1)}') (R12)"
                )

        # R13 (Academy Committee; Farhadi 2013, pp. 38-40): the passive is
        # built on the PRESENT stem + را/رێ — past-stem passives are
        # errors: *نووسترا → نووسرا, *کوشترا → کوژرا, *گرترا → گیرا.
        _PASSIVE_ERRORS = (
            ("نووسترا", "نووسرا"), ("نووسترێ", "نووسرێ"),
            ("کوشترا", "کوژرا"), ("کوشترێ", "کوژرێ"),
            ("فرۆشترا", "فرۆشرا"), ("فرۆشترێ", "فرۆشرێ"),
            ("گرترا", "گیرا"), ("گرترێ", "گیرێ"),
        )
        for word in words:
            for bad, good in _PASSIVE_ERRORS:
                if bad in word:
                    applicable = True
                    violations.append(
                        f"Passive formation: '{word}' — the passive uses "
                        f"the present stem "
                        f"('{word.replace(bad, good, 1)}') (R13)"
                    )
                    break

        # F#164 (Rasul 2004, pp. 125-126): نووسین takes double وو — the
        # under-doubled *نوسی/*دەنوسم family is a frequent spelling error.
        _NUS_FORMS = ("نوسین", "نوسیو", "نوسراو", "دەنوس", "بنوس", "نەنوس", "نوسی")
        for word in words:
            if "نووس" in word:
                continue
            for bad in _NUS_FORMS:
                if bad in word:
                    applicable = True
                    violations.append(
                        f"و/وو spelling: '{word}' — نووسین takes double وو "
                        f"('{word.replace('نوس', 'نووس', 1)}') (F#164)"
                    )
                    break

        # F#161 (Rasul 2005, pp. 35-46): an ئایا question closed with a
        # period is a punctuation error — interrogatives take ؟.
        if "ئایا" in words and sentence.rstrip().endswith("."):
            applicable = True
            violations.append(
                "Interrogative punctuation: ئایا question ends with '.' "
                "instead of '؟' (F#161)"
            )

        # F#119 (Farhadi 2013, pp. 49-51): a short sentence-initial
        # wh-question closed with a period is a punctuation error. Longer
        # sentences are skipped (free relatives: کێ هات پێی بڵێ).
        _WH_INITIAL = ("کێ", "چی", "کوا", "کەی", "بۆچی", "چۆن", "کام")
        content = [w for w in words
                   if w not in {"،", ".", "؟", "!"} and len(w) > 1]
        if (content and content[0] in _WH_INITIAL and len(content) <= 3
                and sentence.rstrip().endswith(".")):
            applicable = True
            violations.append(
                f"Interrogative punctuation: '{content[0]}' question ends "
                f"with '.' instead of '؟' (F#119)"
            )

        # F#36 (Academy Committee, pp. 96-107): suppletive causative
        # pairs — the اندن template never applies to these bases.
        for word in words:
            for bad, good in _SUPPLETIVE_CAUSATIVES:
                if bad in word:
                    applicable = True
                    violations.append(
                        f"Causative formation: '{word}' — this verb has a "
                        f"suppletive causative "
                        f"('{word.replace(bad, good, 1)}') (F#36)"
                    )
                    break

        # F#168 (Rasul, p. 25): خواردن keeps خۆ/خوا in the present —
        # *دەخوێم is a regularisation error (base form خوەردن).
        for word in words:
            good = _XWARDIN_ERRORS.get(word)
            if good is not None:
                applicable = True
                violations.append(
                    f"خواردن stem: '{word}' — the present stem is خۆ/خوا "
                    f"('{good}') (F#168)"
                )
        return applicable, violations

    _NEG_MARKERS = {"نە", "نا", "هیچ", "هەرگیز"}

    def _check_negative_concord(self, sentence: str) -> tuple[bool, list[str]]:
        """Check negation morphology and negative concord.

        Rules enforced:
        - R14 (Academy Committee): نە/مە REPLACE the subjunctive/imperative
          ب prefix — they never co-occur (*نەبچم → نەچم).
        - F#157 (Rasul 2005, pp. 38-41): prohibitive مە is restricted to
          2nd person; 1st-person forms take نە (*مەنووسم → نەنووسم).
        - Negative concord: NPIs like هیچ (nothing) and هەرگیز (never)
          require a negated verb in the same clause (*هیچ دەزانم).

        Applicable when an NPI or a flagged negation form is present.
        """
        violations = []
        applicable = False
        words = self._analyzer.tokenize(sentence)

        # R14: نەب/مەب + present stem = illegal prefix stacking. بوون
        # forms (نەبم, نەبێت) and ب-initial roots (نەبەم, نەبینم) never
        # match because the scan requires a non-ب stem after the ب.
        for word in words:
            for neg in ("نە", "مە"):
                if not word.startswith(neg + "ب"):
                    continue
                rest = word[len(neg) + 1:]
                if any(rest.startswith(s) and len(rest) >= len(s)
                       for s in _PRESENT_STEMS_NON_B):
                    applicable = True
                    violations.append(
                        f"Negation-ب co-occurrence: '{word}' — نە/مە "
                        f"replaces ب ('{neg}{rest}') (R14)"
                    )
                break

        # F#157: مە + stem + 1st-person ending (م/ین) is a person clash.
        for word in words:
            if not word.startswith("مە") or word.startswith("مەب"):
                continue
            rest = word[2:]
            for stem in _PRESENT_STEMS_NON_B:
                if rest.startswith(stem) and rest[len(stem):] in ("م", "ین"):
                    applicable = True
                    violations.append(
                        f"Prohibitive person restriction: '{word}' — مە is "
                        f"2nd-person only; use نە ('نە{rest}') (F#157)"
                    )
                    break

        # F#158 (Rasul 2005, pp. 40-41): the optative negates with نە,
        # never مە (*مەچووبام → نەچووبام). مە is imperative-only. The
        # past-stem guard keeps مە-initial nouns (مەرحەبا) clean.
        _OPTATIVE_ENDINGS = ("بامایە", "بایتایە", "باینایە", "بانایە",
                             "بایە", "باین", "بام", "بای", "بان", "با")
        for word in words:
            if not word.startswith("مە") or len(word) < 5:
                continue
            for end in _OPTATIVE_ENDINGS:
                if not word.endswith(end):
                    continue
                between = word[2: -len(end)]
                if between:
                    bf = self._analyzer.analyze_token(between)
                    if _is_past_verb(between, bf):
                        applicable = True
                        violations.append(
                            f"Optative negation: '{word}' — the optative "
                            f"negates with نە not مە ('نە{word[2:]}') "
                            f"(F#158)"
                        )
                break

        # F#116 (Farhadi 2013, pp. 37-38): under past-progressive negation
        # the Set 1 agent clitic moves BEFORE دە: نەمدەزانی, never
        # *نەدەمزانی. Detection: نەدە + clitic + transitive past stem.
        for word in words:
            if not word.startswith("نەدە"):
                continue
            rest = word[4:]
            for cl in ("مان", "تان", "یان", "م", "ت", "ی"):
                if not rest.startswith(cl):
                    continue
                remainder = rest[len(cl):]
                if remainder and _is_transitive_past(remainder):
                    applicable = True
                    violations.append(
                        f"Negative progressive clitic: '{word}' — the agent "
                        f"clitic precedes دە under negation "
                        f"('نە{cl}دە{remainder}') (F#116)"
                    )
                break

        # F#169 (Rasul, pp. 50-51): نە + ئە assimilate to نا (ە+ە→ا) —
        # the unfused *نەئەچم is a spelling error. Joined demonstratives
        # (نەئەو, نەئەم) are skipped.
        for word in words:
            if word.startswith("نەئە") and len(word) >= 6:
                rest = word[4:]
                if rest and not rest.startswith(("و", "م")):
                    applicable = True
                    violations.append(
                        f"Unfused negation: '{word}' — نە + ئە fuse to نا "
                        f"('نا{rest}') (F#169)"
                    )

        # F#43 (Academy Committee): negation never stacks — *نەنادەچم
        # doubles نە and نا on one verb.
        for word in words:
            if word.startswith("نەنادە") and len(word) >= 8:
                applicable = True
                violations.append(
                    f"Double negation: '{word}' — نە and نا cannot stack "
                    f"('نا{word[6:]}') (F#43)"
                )

        npi_words = {"هیچ", "هەرگیز", "هیچکەس", "هیچکام"}
        has_npi = any(w in npi_words for w in words)
        if not has_npi:
            return applicable, violations
        has_neg_verb = any(
            w.startswith("نا") or w.startswith("نە") for w in words
        )
        if not has_neg_verb:
            npis = [w for w in words if w in npi_words]
            violations.append(
                f"Negative concord: NPI {npis} without negated verb"
            )
        return True, violations

    def _check_conditional_agreement(self, sentence: str) -> tuple[bool, list[str]]:
        """Check conditional clause tense constraints.

        ئەگەر (if) clauses in Sorani typically take subjunctive or past
        tense in the protasis, not indicative present with دە-prefix.
        A sentence like *ئەگەر دەڕۆم is non-standard; the correct form
        uses the bare subjunctive (ئەگەر بچم).

        Applicable when a conditional marker is present.
        """
        violations = []
        applicable = False
        words = self._analyzer.tokenize(sentence)
        cond_markers = {"ئەگەر", "ئەگەری"}
        in_cond = False
        for i, word in enumerate(words):
            if word in cond_markers:
                in_cond = True
                applicable = True
                continue
            if in_cond:
                # End condition at clause boundaries
                if word in {"،", ".", "؟", "!"}:
                    in_cond = False
                    continue
                # دە-prefix in conditional protasis is non-standard
                if word.startswith("دە") and _is_present_verb(word):
                    violations.append(
                        f"Conditional agreement: indicative '{word}' in "
                        f"ئەگەر-clause; expected subjunctive (ب-prefix)"
                    )
                    in_cond = False
        return applicable, violations

    # ── CRIT-4: Four additional agreement checks ──

    # Sorani quantifiers that force plural agreement on the verb
    # (Slevanayi 2001, pp. 87-88; Maaruf 2010, p. 139)
    _QUANTIFIERS_PLURAL = {"هەر", "هیچ", "هەموو", "چەند", "هەندێک"}

    def _check_quantifier_noun(self, sentence: str) -> tuple[bool, list[str]]:
        """Check quantifier–verb number agreement.

        In Central Kurdish (Sorani), certain quantifiers (هەموو, هەر, هیچ, چەند,
        هەندێک) govern a plural verb. For example:
        هەموو منداڵ *دەچێت is wrong; the correct form is
        هەموو منداڵ دەچن (3pl).

        Source: Slevanayi (2001), pp. 87-88; Maaruf (2010), p. 139.

        Applicable when a governing quantifier is paired with a verb.
        """
        violations = []
        applicable = False
        words = self._analyzer.tokenize(sentence)
        clause_bounds = set(self._clause_boundary_indices(words))

        for i, word in enumerate(words):
            if word not in self._QUANTIFIERS_PLURAL:
                continue
            # Scan forward for a verb within the same clause
            for j in range(i + 1, len(words)):
                if j in clause_bounds:
                    break
                candidate = words[j]
                is_present = any(candidate.startswith(p) for p in _PRESENT_PREFIXES)
                is_neg_present = candidate.startswith(_NEGATION_PRESENT_PREFIX)
                if not (is_present or is_neg_present):
                    continue
                verb_pn = self._verb_ending_to_pn(candidate)
                if verb_pn is None:
                    break
                applicable = True
                _, verb_number = verb_pn
                if verb_number == "sg":
                    violations.append(
                        f"Quantifier–verb mismatch: '{word}' requires plural "
                        f"verb, but '{candidate}' is singular"
                    )
                break

        # F#77 (Slevanayi 2001, pp. 87-88): numeral subjects force a
        # plural verb: دوو کوڕ هاتن — *دوو کوڕ هات. Restricted to
        # adjacent intransitive verbs so numeral OBJECTS (دوو سێو
        # دەخوات) and duration adverbials (دوو ڕۆژ مایەوە) stay clean.
        for i, word in enumerate(words):
            if word not in _NUMERALS_PLURAL or i + 2 >= len(words):
                continue
            if i > 0 and words[i - 1] not in _SUBJ_CONTEXT_OK:
                continue
            noun, verb = words[i + 1], words[i + 2]
            if noun in _TIME_NOUNS:
                continue
            flagged_sg = False
            good = ""
            for stem in _PAST_INTRANS_STEMS_CHECK:
                if verb.startswith(stem):
                    sfx = verb[len(stem):]
                    if sfx in _PAST_SET2_SUFFIXES:
                        applicable = True
                        flagged_sg = sfx == ""
                        good = f"{stem}ن"
                    break
            if not flagged_sg and not good:
                for pre in ("دە", "ئە", "نا"):
                    if not verb.startswith(pre):
                        continue
                    rest = verb[len(pre):]
                    for stem in _INTRANS_PRESENT_STEMS + ("گەڕ",):
                        if rest.startswith(stem) and rest[len(stem):] in (
                                "ێت", "ات", "ێ", "ێتەوە", "اتەوە"):
                            applicable = True
                            flagged_sg = True
                            good = f"{pre}{stem}ن"
                            break
                    break
            if flagged_sg:
                violations.append(
                    f"Numeral subject: '{word} {noun}' requires plural "
                    f"verb but '{verb}' is singular ('{good}') (F#77)"
                )
        return applicable, violations

    # Sorani relative clause markers
    _REL_MARKERS = {"کە", "ئەوەی"}

    def _check_relative_clause(self, sentence: str) -> tuple[bool, list[str]]:
        """Check antecedent–verb agreement in relative clauses.

        When a relative clause (introduced by کە or ئەوەی) modifies an
        antecedent, the verb inside the relative clause should agree in
        person and number with the antecedent head noun—not with any
        intervening NP.

        Heuristic: if the word before کە is a pronoun with known person/
        number, the first verb after کە should agree with it.

        Applicable when a relative marker has a determinable antecedent
        and an inner verb.
        """
        violations = []
        applicable = False
        words = self._analyzer.tokenize(sentence)

        for i, word in enumerate(words):
            if word not in self._REL_MARKERS:
                continue
            # Antecedent is the previous word (head noun of the NP)
            if i == 0:
                continue
            antecedent = words[i - 1]
            ant_person: str | None = None
            ant_number: str | None = None
            if antecedent in _PRONOUN_AGREEMENT:
                ant_person, ant_number = _PRONOUN_AGREEMENT[antecedent]
            elif antecedent.endswith("ەکان") or antecedent.endswith("یەکان"):
                ant_person, ant_number = "3", "pl"
            elif antecedent.endswith("ەکە") or antecedent.endswith("یەکە"):
                ant_person, ant_number = "3", "sg"
            else:
                continue  # cannot determine antecedent features

            # Scan for the first verb inside the relative clause
            for j in range(i + 1, len(words)):
                if words[j] in {"،", ".", "؟", "!"}:
                    break
                candidate = words[j]
                is_present = any(candidate.startswith(p) for p in _PRESENT_PREFIXES)
                is_neg_present = candidate.startswith(_NEGATION_PRESENT_PREFIX)
                if not (is_present or is_neg_present):
                    continue
                verb_pn = self._verb_ending_to_pn(candidate)
                if verb_pn is None:
                    break
                applicable = True
                verb_person, verb_number = verb_pn
                if ant_number and verb_number != ant_number:
                    violations.append(
                        f"Relative clause number mismatch: antecedent '{antecedent}' "
                        f"({ant_person}{ant_number}) but verb '{candidate}' "
                        f"({verb_person}{verb_number}) in کە-clause"
                    )
                if ant_person and verb_person != ant_person:
                    violations.append(
                        f"Relative clause person mismatch: antecedent '{antecedent}' "
                        f"({ant_person}{ant_number}) but verb '{candidate}' "
                        f"({verb_person}{verb_number}) in کە-clause"
                    )
                break
        return applicable, violations

    # Vocative markers and imperative detection
    _VOCATIVE_MARKERS = {"ئەی", "یا"}

    def _check_vocative_imperative(self, sentence: str) -> tuple[bool, list[str]]:
        """Check vocative marker–imperative verb number agreement.

        A sentence beginning with a vocative marker (ئەی, یا) followed
        by a singular addressee should have a 2sg imperative; with a
        plural addressee (or plural noun), 2pl imperative.

        Imperative verbs in Sorani start with ب- (or بی-).

        Applicable when a vocative-led clause has a determinable addressee
        number and an imperative verb.
        """
        violations = []
        applicable = False
        words = self._analyzer.tokenize(sentence)
        if not words:
            return False, violations

        # R15/F#42 (Academy Committee, pp. 182-191): the 2sg imperative of
        # a consonant-final present stem requires final ە (بنووسە, مەگرە);
        # a bare *بنووس/*مەگر is an incomplete imperative.
        for word in words:
            if word == "بچە":
                # F#42 exception: the imperative of چوون is بچۆ (irregular).
                applicable = True
                violations.append(
                    "Imperative of چوون: 'بچە' — the standard form is "
                    "'بچۆ' (F#42)"
                )
                continue
            for pre in ("ب", "مە"):
                if not word.startswith(pre):
                    continue
                if word[len(pre):] in _IMPERATIVE_E_STEMS:
                    applicable = True
                    violations.append(
                        f"Imperative missing ە: '{word}' — the 2sg "
                        f"imperative of a consonant-final stem requires ە "
                        f"('{word}ە') (R15/F#42)"
                    )
                break

        # F#125 (Haji Marf 2014, p. 192): intransitive imperatives NEVER
        # host a Set 1 clitic — *بمکەوە/*مەتچۆ are ungrammatical, while
        # transitive بمگرە (catch me!) is fine. Causatives (بیخەوێنە)
        # are transitive and exempt via the ێن guard.
        for word in words:
            for pre in ("ب", "مە"):
                if not word.startswith(pre):
                    continue
                rest = word[len(pre):]
                for cl in ("مان", "تان", "یان", "م", "ت", "ی"):
                    if not rest.startswith(cl):
                        continue
                    stem_part = rest[len(cl):]
                    if ("ێن" not in stem_part
                            and any(stem_part.startswith(s)
                                    for s in _INTRANS_PRESENT_STEMS)):
                        applicable = True
                        violations.append(
                            f"Imperative clitic restriction: '{word}' — "
                            f"intransitive imperatives never host a Set 1 "
                            f"clitic (F#125)"
                        )
                    break
                break

        # F#76 (Slevanayi 2001, pp. 16, 72-73): a plural vocative (ینە)
        # demands a plural imperative: کوڕینە بڕۆن — *کوڕینە بڕۆ.
        if words[0] in _PLURAL_VOCATIVES:
            for word in words[1:]:
                imp_stemmed = (
                    (word.startswith("ب") and any(
                        word[1:].startswith(s) for s in _PRESENT_STEMS_NON_B))
                    or (word.startswith("مە") and any(
                        word[2:].startswith(s) for s in _PRESENT_STEMS_NON_B))
                )
                if not imp_stemmed:
                    continue
                applicable = True
                if not word.endswith("ن"):
                    fix = word[:-1] + "ن" if word.endswith("ە") else word + "ن"
                    violations.append(
                        f"Vocative-imperative mismatch: plural vocative "
                        f"'{words[0]}' but imperative '{word}' is singular "
                        f"('{fix}') (F#76)"
                    )
                break

        if words[0] not in self._VOCATIVE_MARKERS:
            return applicable, violations

        # Determine addressee number from the noun after vocative marker
        addressee_number: str | None = None
        for k in range(1, min(len(words), 4)):
            w = words[k]
            if w.endswith("ەکان") or w.endswith("یەکان"):
                addressee_number = "pl"
                break
            elif w.endswith("ەکە") or w.endswith("یەکە"):
                addressee_number = "sg"
                break
            elif w in _PLURAL_VOCATIVES:
                addressee_number = "pl"
                break
            elif w in _PRONOUN_AGREEMENT:
                _, addressee_number = _PRONOUN_AGREEMENT[w]
                break

        if addressee_number is None:
            return applicable, violations

        # Find imperative verb (ب-prefix)
        for word in words:
            if word.startswith("ب") and len(word) > 2 and not _is_present_verb(word):
                applicable = True
                # Imperative 2sg typically ends without ن; 2pl ends with ن
                if addressee_number == "pl" and not word.endswith("ن"):
                    violations.append(
                        f"Vocative-imperative mismatch: plural addressee "
                        f"but imperative '{word}' is singular"
                    )
                elif addressee_number == "sg" and word.endswith("ن"):
                    violations.append(
                        f"Vocative-imperative mismatch: singular addressee "
                        f"but imperative '{word}' is plural"
                    )
                break
        return applicable, violations

    # Temporal adverbs with tense constraints
    _PAST_ADVERBS = {"دوێنێ", "پار", "پێشتر", "بەرلە", "پارێ"}
    _PRESENT_ADVERBS = {"ئێستا", "ئەمڕۆ", "دواتر", "هەمیشە"}

    def _check_adverb_verb_tense(self, sentence: str) -> tuple[bool, list[str]]:
        """Check temporal adverb–verb tense consistency.

        Temporal adverbs constrain the tense of the clause verb. A past
        adverb (دوێنێ = yesterday, پار = last year) with a present-tense
        verb is inconsistent, and vice versa.

        Applicable when a temporal adverb co-occurs with a detectable
        clause tense.
        """
        violations = []
        words = self._analyzer.tokenize(sentence)
        has_past_adv = any(w in self._PAST_ADVERBS for w in words)
        has_present_adv = any(w in self._PRESENT_ADVERBS for w in words)

        if not (has_past_adv or has_present_adv):
            return False, violations

        clause_tense = self._detect_clause_tense(words)
        if clause_tense is None:
            return False, violations

        if has_past_adv and clause_tense == "present":
            adverbs = [w for w in words if w in self._PAST_ADVERBS]
            violations.append(
                f"Adverb-tense mismatch: past adverb(s) {adverbs} "
                f"with present-tense verb"
            )
        if has_present_adv and clause_tense == "past":
            adverbs = [w for w in words if w in self._PRESENT_ADVERBS]
            violations.append(
                f"Adverb-tense mismatch: present adverb(s) {adverbs} "
                f"with past-tense verb"
            )
        return True, violations

    # Person hierarchy for compound subjects: 1st > 2nd > 3rd
    _PERSON_HIERARCHY = {"1": 3, "2": 2, "3": 1}

    def _check_compound_subject(self, sentence: str) -> tuple[bool, list[str]]:
        """Check compound subject person resolution (Slevanayi 2001, p. 89).

        When two subjects are coordinated with و, the verb should agree with
        the highest person in the hierarchy: 1st > 2nd > 3rd.
        Example: من و تۆ دەچین (I and you go-1pl), NOT *من و تۆ دەچن (go-3pl).

        Applicable when at least two pronouns are coordinated with و.
        """
        violations = []
        words = self._analyzer.tokenize(sentence)

        # F#88 (Slevanayi 2001, p. 61): compound NOUN subjects (X و Y) always
        # force a plural verb — کچ و کوڕ هاتن, never *کچ و کوڕ هات.
        # Conservative pattern: sentence-initial N و N immediately followed by
        # an intransitive/present verb. Past transitives are excluded (Law 2:
        # the verb agrees with the object there), and a clitic-hosting second
        # noun is excluded (the pair is then an object, e.g. کتێب و قەڵەمم کڕی).
        noun_coord_applicable = False
        if len(words) >= 4 and words[1] == "و":
            f0 = self._analyzer.analyze_token(words[0])
            f2 = self._analyzer.analyze_token(words[2])
            verb_tok = words[3]
            fv = self._analyzer.analyze_token(verb_tok)
            second_hosts_clitic = bool(getattr(f2, "is_clitic", False))
            both_nominal = (
                f0.pos in ("NOUN", "PROPN") and f2.pos in ("NOUN", "PROPN")
            )
            # Isolated tokens like هات analyze as NOUN; use the builder's
            # past-verb detector (same one the graph builder relies on).
            is_verb = (
                fv.pos == "VERB"
                or _is_present_verb(verb_tok)
                or _is_past_verb(verb_tok, fv)
            )
            if (both_nominal and not second_hosts_clitic and is_verb
                    and not _is_transitive_past(verb_tok)):
                noun_coord_applicable = True
                # Plural if overt 3pl marking (ن-final finite or infinitive
                # homograph like هاتن reinterpreted as past 3pl); else singular.
                is_plural = (
                    fv.number == "pl"
                    or fv.tense == "infinitive"
                    or verb_tok.endswith("ن")
                )
                if not is_plural:
                    violations.append(
                        f"Compound noun subject: '{words[0]} و {words[2]}' "
                        f"requires plural verb but '{verb_tok}' is singular (F#88)"
                    )

        # F#87 (Slevanayi 2001, pp. 68-69): coordinated subjects follow
        # the familiarity hierarchy 1st > 2nd > 3rd: من و ئازاد, never
        # *ئازاد و من. Clause coordination (verb before و) is skipped.
        for i, word in enumerate(words):
            if word != "و" or i == 0 or i + 1 >= len(words):
                continue
            left, right = words[i - 1], words[i + 1]
            r_rank = _FAMILIARITY_RANK.get(right)
            if r_rank is None:
                continue
            l_rank = _FAMILIARITY_RANK.get(left)
            if l_rank is None:
                lf = self._analyzer.analyze_token(left)
                if (lf.pos == "VERB" or _is_present_verb(left)
                        or _is_past_verb(left, lf)):
                    continue
                if lf.pos not in ("NOUN", "PROPN"):
                    continue
                l_rank = 3
            noun_coord_applicable = True
            if l_rank > r_rank:
                violations.append(
                    f"Familiarity order: '{left} و {right}' — coordination "
                    f"follows 1st > 2nd > 3rd ('{right} و {left}') (F#87)"
                )

        # Find coordinated pronoun subjects: pronoun + و + pronoun
        pronouns_found = []
        for i, word in enumerate(words):
            if word in _PRONOUN_AGREEMENT:
                pronouns_found.append((i, word))

        if len(pronouns_found) < 2:
            return noun_coord_applicable, violations

        # Check if pronouns are coordinated with و
        coordinated_pronouns = []
        for idx in range(len(pronouns_found) - 1):
            pos_a = pronouns_found[idx][0]
            pos_b = pronouns_found[idx + 1][0]
            # Check for و between the two pronouns
            if pos_b - pos_a == 2 and pos_a + 1 < len(words) and words[pos_a + 1] == "و":
                coordinated_pronouns.append(pronouns_found[idx][1])
                coordinated_pronouns.append(pronouns_found[idx + 1][1])

        if len(coordinated_pronouns) < 2:
            return noun_coord_applicable, violations

        applicable = True
        # Determine expected person: highest in hierarchy
        persons = [_PRONOUN_AGREEMENT[p][0] for p in coordinated_pronouns]
        expected_person = max(persons, key=lambda p: self._PERSON_HIERARCHY.get(p, 0))

        # Find the verb after the compound subject
        last_pronoun_pos = max(pos for pos, _ in pronouns_found if _ in coordinated_pronouns)
        for j in range(last_pronoun_pos + 1, min(last_pronoun_pos + 8, len(words))):
            word = words[j]
            for ending, (person, _number) in _PRESENT_ENDINGS.items():
                if word.endswith(ending) and any(word.startswith(p) for p in _PRESENT_PREFIXES):
                    if person != expected_person:
                        violations.append(
                            "Compound subject person mismatch: "
                            f"coordinated pronouns {coordinated_pronouns} "
                            f"expect person={expected_person} but verb "
                            f"'{word}' has person={person}"
                        )
                    return applicable, violations

        return applicable, violations

    # Common bare nouns (non-pronominal) — treated as 3sg for agreement
    _BARE_NOUN_INDICATORS = {"پیاو", "ژن", "منداڵ", "مامۆستا", "قوتابی", "کچ", "کوڕ"}

    def _check_bare_noun_agreement(self, sentence: str) -> tuple[bool, list[str]]:
        """Check bare noun person-only agreement (Slevanayi 2001, p. 60).

        Bare nouns (without demonstrative/definite marker) agree with the
        verb in person only (3sg) without number constraint. A 1st or 2nd
        person verb after a bare noun subject is a violation.

        Applicable when a bare noun indicator is paired with a verb.
        """
        violations = []
        applicable = False
        words = self._analyzer.tokenize(sentence)

        for i, word in enumerate(words):
            if word not in self._BARE_NOUN_INDICATORS:
                continue
            # Look ahead for a verb, skip if a demonstrative precedes
            if i > 0 and words[i - 1] in ("ئەو", "ئەم", "ئەوان"):
                continue

            for j in range(i + 1, min(i + 8, len(words))):
                v = words[j]
                if v == "و" or v in ("،", ".", "؟", "!"):
                    break
                for ending, (person, _number) in _PRESENT_ENDINGS.items():
                    if v.endswith(ending) and any(v.startswith(p) for p in _PRESENT_PREFIXES):
                        applicable = True
                        if person in ("1", "2"):
                            violations.append(
                                f"Bare noun '{word}' expects 3rd person verb "
                                f"but '{v}' has person={person}"
                            )
                        return applicable, violations

        return applicable, violations


    def _check_noun_subject_verb_number(self, sentence: str) -> tuple[bool, list[str]]:
        """Check number agreement between a non-pronominal subject and its verb (Law 1).

        Complements ``_check_subject_verb`` (pronoun + present-tense verb) and
        ``_check_bare_noun_agreement`` (bare noun, person-only) by covering
        definite/proper-noun subjects whose verb must match them in number
        under Law 1 (Slevanayi 2001, pp. 60, 89). Works across tenses,
        including past intransitives such as the چوون/چوو homograph, which the
        agreement graph disambiguates to a finite past form in verb position.

        Applicable only when the graph builds a subject-verb (Law 1) edge whose
        non-pronominal subject and verb both carry an explicit number.
        """
        violations: list[str] = []
        applicable = False
        try:
            graph = build_agreement_graph(sentence, self._analyzer)
        except (KeyError, AttributeError, TypeError, IndexError) as exc:
            logger.debug("noun-subject number check skipped: %s", exc)
            return False, []

        _SUBJ_VERB_LAW1 = {
            "subject_verb", "passive_subject_verb", "backward_subject_verb",
        }
        for e in graph.edges:
            if e.agreement_type not in _SUBJ_VERB_LAW1:
                continue
            if e.source_idx >= len(graph.features) or e.target_idx >= len(graph.features):
                continue
            subj = graph.features[e.source_idx]
            verb = graph.features[e.target_idx]
            # Pronoun subjects are handled by _check_subject_verb; skip to
            # avoid double-counting the same relation.
            if subj.pos == "PRON":
                continue
            # Infinitives (verbal nouns) are not finite verbs and take no
            # subject-verb number agreement (e.g. «چوون قورسە» = "going is
            # hard"). Only the finite past reading, disambiguated by the
            # builder in verb position, is checked.
            if verb.tense == "infinitive":
                continue
            if not subj.number or not verb.number:
                continue
            applicable = True
            if subj.number != verb.number:
                violations.append(
                    f"Subject-verb number mismatch: "
                    f"'{graph.tokens[e.source_idx]}' ({subj.number}) with verb "
                    f"'{graph.tokens[e.target_idx]}' ({verb.number})"
                )
        return applicable, violations


def evaluate_agreement_accuracy(
    sentences: list[str],
    checker: Optional[AgreementChecker] = None,
) -> dict:
    """Compute agreement accuracy over a corpus.

    Reports two denominators. ``accuracy`` keeps the legacy sentence-level
    pass rate (fraction of sentences with no violation), which is inflated
    on corpora full of sentences where no check applies. ``accuracy_applicable``
    restricts to sentences with at least one applicable check — the
    denominator the metric was meant to use.

    Returns:
        Dict with accuracy fields, totals, and average check counts.
    """
    if checker is None:
        checker = AgreementChecker()

    results = [checker.check_sentence(s) for s in sentences]

    correct = sum(1 for r in results if r.is_correct)
    total = len(results)

    applicable_results = [r for r in results if r.is_applicable]
    n_applicable = len(applicable_results)
    correct_applicable = sum(1 for r in applicable_results if r.is_correct)

    return {
        "accuracy": correct / total if total > 0 else 0.0,
        "accuracy_applicable": (
            correct_applicable / n_applicable if n_applicable > 0 else 0.0
        ),
        "total_sentences": total,
        "correct_sentences": correct,
        "applicable_sentences": n_applicable,
        "correct_applicable_sentences": correct_applicable,
        "avg_checks_passed": sum(r.checks_passed for r in results) / total if total > 0 else 0,
        "avg_checks_total": sum(r.checks_total for r in results) / total if total > 0 else 0,
        "avg_checks_applicable": (
            sum(r.checks_applicable for r in results) / total if total > 0 else 0
        ),
    }


# PIPE-4: Per-agreement-law breakdown — labels match check_sentence order
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


def evaluate_agreement_by_check(
    sentences: list[str],
    checker: Optional[AgreementChecker] = None,
) -> dict[str, dict]:
    """Per-check accuracy breakdown, returning stats for each of the 15 checks.

    Also aggregates Law 1 (subject-verb) and Law 2 (object-verb ergative)
    separately — the two agreement laws central to this thesis.
    """
    if checker is None:
        checker = AgreementChecker()

    per_check: dict[str, dict] = {}
    for label, _law in _CHECK_LABELS:
        per_check[label] = {"correct": 0, "total": 0, "law": _law}

    check_methods = [
        "_check_subject_verb",
        "_check_clitic_consistency",
        "_check_ezafe",
        "_check_tense_consistency",
        "_check_object_verb_ergative",
        "_check_negative_concord",
        "_check_orthography",
        "_check_conditional_agreement",
        "_check_quantifier_noun",
        "_check_relative_clause",
        "_check_vocative_imperative",
        "_check_adverb_verb_tense",
        "_check_compound_subject",
        "_check_bare_noun_agreement",
        "_check_noun_subject_verb_number",
    ]

    for sent in sentences:
        for (label, _law), method_name in zip(_CHECK_LABELS, check_methods):
            method = getattr(checker, method_name)
            applicable, violations = method(sent)
            # Only count a check toward its denominator when it applied —
            # inapplicable checks are not silent passes.
            if not applicable:
                continue
            per_check[label]["total"] += 1
            if not violations:
                per_check[label]["correct"] += 1

    # Compute accuracy per check
    for label in per_check:
        t = per_check[label]["total"]
        c = per_check[label]["correct"]
        per_check[label]["accuracy"] = c / t if t > 0 else 0.0

    # Aggregate Law 1 / Law 2
    law_summary = {}
    for label, info in per_check.items():
        law = info.get("law", "")
        if law:
            if law not in law_summary:
                law_summary[law] = {"correct": 0, "total": 0}
            law_summary[law]["correct"] += info["correct"]
            law_summary[law]["total"] += info["total"]
    for law in law_summary:
        t = law_summary[law]["total"]
        c = law_summary[law]["correct"]
        law_summary[law]["accuracy"] = c / t if t > 0 else 0.0

    return {"per_check": per_check, "per_law": law_summary}
