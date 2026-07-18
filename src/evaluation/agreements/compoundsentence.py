from typing import Optional
from .constants import *

class CompoundSentenceMixin:
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
        expected_person = max(persons, key=lambda p: _PERSON_HIERARCHY.get(p, 0))
    
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

    def _check_cross_clause_covert_subject(self, sentence: str) -> tuple[bool, list[str]]:
        violations = []
        applicable = False
        words = self._analyzer.tokenize(sentence)
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

