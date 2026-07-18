from typing import Optional
from .constants import *

class SimpleSentenceMixin:
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
            for a, b in _ORTHO_CONFUSIONS:
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
            if word not in _QUANTIFIERS_PLURAL:
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
    
        if words[0] not in _VOCATIVE_MARKERS:
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
        has_past_adv = any(w in _PAST_ADVERBS for w in words)
        has_present_adv = any(w in _PRESENT_ADVERBS for w in words)
    
        if not (has_past_adv or has_present_adv):
            return False, violations
    
        clause_tense = self._detect_clause_tense(words)
        if clause_tense is None:
            return False, violations
    
        if has_past_adv and clause_tense == "present":
            adverbs = [w for w in words if w in _PAST_ADVERBS]
            violations.append(
                f"Adverb-tense mismatch: past adverb(s) {adverbs} "
                f"with present-tense verb"
            )
        if has_present_adv and clause_tense == "past":
            adverbs = [w for w in words if w in _PRESENT_ADVERBS]
            violations.append(
                f"Adverb-tense mismatch: present adverb(s) {adverbs} "
                f"with past-tense verb"
            )
        return True, violations

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
            if word not in _BARE_NOUN_INDICATORS:
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

