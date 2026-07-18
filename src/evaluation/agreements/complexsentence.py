from typing import Optional
from .constants import *

class ComplexSentenceMixin:
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
            if word not in _REL_MARKERS:
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

