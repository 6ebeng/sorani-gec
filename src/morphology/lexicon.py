"""Standalone Sorani Kurdish Lexicon Tool

A self-contained morphological lexicon for Sorani Kurdish, built by
parsing both the .dic (dictionary) and .aff (affix rules) files from
the KurdishHunspell project into native Python data structures.

This removes any runtime dependency on the Hunspell library or format,
providing:
  - Word validation (spell checking) via affix rule application
  - Verb stem lookup with transitivity detection
  - Morphological decomposition (surface form -> stem + affixes)
  - Surface form generation (stem + flag -> inflected forms)

Based on KurdishHunspell v0.1.1 (Ahmadi, 2020-2025).
33,856 dictionary entries, ~5,400 affix rules covering nouns,
adjectives, adverbs, verbs (present/past, transitive/intransitive),
passive voice, numerals, and particles.

Flag inventory (from .aff header):
    N  Noun                 A  Adjective            R  Adverb
    V  Present intransitive M  Present transitive   P  Passive
    I  Past intransitive    T  Past transitive      E  Numeral
    G  Particle             B  Pronoun              C  Conjunction
    D  Interjection         F  Adposition           X  Infinitive
    Z  Proper name          W  Verb exceptional     H  Punctuation
    J  Digit (compound)
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ── Data Classes ─────────────────────────────────────────────────


@dataclass
class LexiconEntry:
    """A single dictionary entry parsed from the .dic file."""

    surface: str
    flags: str
    pos: str
    stem: str
    lemma: str
    inflectional: str


@dataclass
class AffixRule:
    """A single prefix or suffix rule parsed from the .aff file."""

    kind: str  # "PFX" or "SFX"
    flag: str  # Hunspell flag character
    strip: str  # Characters to strip (empty = none)
    add: str  # Characters to add (empty = none)
    condition: re.Pattern  # Compiled regex for stem matching


@dataclass
class MorphAnalysis:
    """Result of morphological decomposition of a surface form."""

    surface: str
    stem: str
    lemma: str
    pos: str
    flags: str
    prefix: str = ""
    suffix: str = ""
    inflectional: str = ""


# ── Main Lexicon Class ───────────────────────────────────────────


class SoraniLexicon:
    """Standalone Sorani Kurdish lexicon and morphological tool.

    Parses both .dic and .aff files into native Python data structures,
    replacing any runtime dependency on the Hunspell library or format.

    The .dic file provides 33,856 surface forms with POS tags, stems,
    lemmas, and inflectional status. The .aff file provides ~5,400
    affix rules (prefix and suffix) organized by flag, encoding the
    full inflectional morphology of Sorani Kurdish: noun declension,
    adjective comparison, verb conjugation (4 tense/transitivity
    paradigms), passive voice, and more.
    """

    FLAG_POS = {
        "N": "noun",
        "A": "adj",
        "R": "adv",
        "V": "verb_pres_intrans",
        "M": "verb_pres_trans",
        "I": "verb_past_intrans",
        "T": "verb_past_trans",
        "P": "passive",
        "E": "numeral",
        "G": "particle",
        "B": "pron",
        "C": "conj",
        "D": "intj",
        "F": "adp",
        "X": "infinitive",
        "Z": "propn",
        "W": "verb_exceptional",
        "H": "punct",
        "J": "digit",
    }

    VERB_FLAGS = {"V", "M", "I", "T", "W"}
    TRANSITIVE_FLAGS = {"T", "M"}
    INTRANSITIVE_FLAGS = {"I", "V"}

    def __init__(
        self,
        dic_path: Optional[str] = None,
        aff_path: Optional[str] = None,
    ):
        self.entries: dict[str, list[LexiconEntry]] = {}
        self.words: set[str] = set()
        self.prefix_rules: dict[str, list[AffixRule]] = {}
        self.suffix_rules: dict[str, list[AffixRule]] = {}
        self.cross_product: dict[str, bool] = {}
        self.replacements: list[tuple[str, str]] = []
        self.try_chars: str = ""
        self.ignore_chars: str = ""
        self.compound_flag: str = ""
        self._ignore_set: set[str] = set()
        self.available = False

        self._resolve_and_load(dic_path, aff_path)

    # ── Loading ──────────────────────────────────────────────────

    def _resolve_and_load(
        self, dic_path: Optional[str], aff_path: Optional[str]
    ):
        """Resolve file paths and load both .dic and .aff resources."""
        base = Path(__file__).resolve().parent.parent.parent / "data"

        if dic_path is None:
            for candidate in [
                base / "hunspell" / "ckb-Arab.dic",
                base / "lexicon" / "ckb-Arab.dic",
            ]:
                if candidate.is_file():
                    dic_path = str(candidate)
                    break

        if dic_path is None or not Path(dic_path).is_file():
            logger.warning(
                "Dictionary file not found. Lexicon unavailable."
            )
            return

        if aff_path is None:
            candidate = base / "hunspell" / "ckb-Arab.aff"
            if candidate.is_file():
                aff_path = str(candidate)

        self._load_dic(dic_path)

        if aff_path and Path(aff_path).is_file():
            self._load_aff(aff_path)
        else:
            logger.info("No affix file found; dictionary-only mode.")

        self._ignore_set = set(self.ignore_chars)
        self.available = True

        logger.info(
            "SoraniLexicon: %d words, %d PFX rules, %d SFX rules",
            len(self.words),
            sum(len(v) for v in self.prefix_rules.values()),
            sum(len(v) for v in self.suffix_rules.values()),
        )

    def _load_dic(self, path: str):
        """Parse dictionary entries from the .dic file."""
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # First line is the entry count; skip it
        if lines and lines[0].strip().isdigit():
            lines = lines[1:]

        for line in lines:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if not parts:
                continue
            word_part = parts[0]

            surface = word_part
            flags = ""
            if "/" in word_part:
                surface, flags = word_part.split("/", 1)

            self.words.add(surface)

            pos = stem = lemma = inflectional = ""
            for tag in parts[1:]:
                if tag.startswith("po:"):
                    pos = tag[3:]
                elif tag.startswith("st:"):
                    stem = tag[3:]
                elif tag.startswith("lem:"):
                    lemma = tag[4:]
                elif tag.startswith("is:"):
                    inflectional = tag[3:]

            entry = LexiconEntry(
                surface=surface,
                flags=flags,
                pos=pos,
                stem=stem,
                lemma=lemma,
                inflectional=inflectional,
            )
            self.entries.setdefault(surface, []).append(entry)

    def _load_aff(self, path: str):
        """Parse affix rules and metadata from the .aff file."""
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if not parts:
                continue

            keyword = parts[0]

            # ── Metadata directives ──
            if keyword == "TRY" and len(parts) > 1:
                self.try_chars = parts[1]
                continue
            if keyword == "IGNORE" and len(parts) > 1:
                self.ignore_chars = parts[1]
                continue
            if keyword == "COMPOUNDFLAG" and len(parts) > 1:
                self.compound_flag = parts[1]
                continue
            if keyword == "REP" and len(parts) == 3:
                self.replacements.append((parts[1], parts[2]))
                continue

            # ── Affix rules ──
            if keyword not in ("PFX", "SFX") or len(parts) < 4:
                continue

            flag = parts[1]

            # Header line: PFX/SFX flag Y/N count
            if parts[2] in ("Y", "N") and parts[3].isdigit():
                self.cross_product[flag] = parts[2] == "Y"
                continue

            # Rule line: PFX/SFX flag strip add [condition]
            strip = parts[2] if parts[2] != "0" else ""
            add = parts[3] if parts[3] != "0" else ""
            condition_str = parts[4] if len(parts) > 4 else "."

            try:
                if keyword == "SFX":
                    pattern = re.compile(condition_str + "$")
                else:
                    pattern = re.compile("^" + condition_str)
            except re.error:
                continue

            rule = AffixRule(
                kind=keyword,
                flag=flag,
                strip=strip,
                add=add,
                condition=pattern,
            )

            if keyword == "PFX":
                self.prefix_rules.setdefault(flag, []).append(rule)
            else:
                self.suffix_rules.setdefault(flag, []).append(rule)

    # ── Normalization ────────────────────────────────────────────

    def normalize(self, word: str) -> str:
        """Strip ignored diacritical marks from a word."""
        if not self._ignore_set:
            return word
        return "".join(c for c in word if c not in self._ignore_set)

    # ── Spell Checking ───────────────────────────────────────────

    def is_valid(self, word: str) -> bool:
        """Check if a word is valid: in dictionary or decomposable
        through affix rules."""
        word = self.normalize(word)
        if word in self.words:
            return True
        return len(self._find_analyses(word, first_only=True)) > 0

    def is_correct(self, word: str) -> bool:
        """Backward-compatible alias for is_valid."""
        return self.is_valid(word)

    def suggest(self, word: str) -> list[str]:
        """Generate spelling correction candidates via REP rules."""
        candidates = []
        for pattern, replacement in self.replacements:
            try:
                fixed = re.sub(pattern, replacement, word)
                if fixed != word and fixed in self.words:
                    candidates.append(fixed)
            except re.error:
                continue
        return candidates

    # ── Verb Stem Lookup ─────────────────────────────────────────

    def find_verb_stem(
        self, text: str
    ) -> Optional[tuple[str, LexiconEntry]]:
        """Longest-prefix match for verb stems in the dictionary.

        Used by MorphologicalAnalyzer for transitivity detection.
        Backward-compatible with AhmadiLexiconParser.find_verb_stem().
        """
        if not self.available:
            return None

        for length in range(len(text), 0, -1):
            candidate = text[:length]
            if candidate in self.entries:
                for entry in self.entries[candidate]:
                    if self._is_verb_entry(entry):
                        return candidate, entry
        return None

    # ── Morphological Decomposition ──────────────────────────────

    def decompose(self, word: str) -> list[MorphAnalysis]:
        """Decompose a surface form into all valid stem + affix analyses.

        Tries suffix-only, prefix-only, and prefix+suffix (cross-product)
        strippings against the dictionary.
        """
        return self._find_analyses(self.normalize(word))

    def _find_analyses(
        self, word: str, first_only: bool = False
    ) -> list[MorphAnalysis]:
        """Core analysis: find morphological decompositions for a word."""
        results: list[MorphAnalysis] = []

        # 1. Direct dictionary match
        if word in self.entries:
            for entry in self.entries[word]:
                results.append(
                    MorphAnalysis(
                        surface=word,
                        stem=entry.stem or word,
                        lemma=entry.lemma or word,
                        pos=entry.pos,
                        flags=entry.flags,
                        inflectional=entry.inflectional,
                    )
                )
                if first_only:
                    return results

        # 2. Suffix-only stripping
        for flag, rules in self.suffix_rules.items():
            for rule in rules:
                stem = self._try_strip_suffix(word, rule)
                if stem is None or stem not in self.entries:
                    continue
                for entry in self.entries[stem]:
                    if flag in entry.flags:
                        results.append(
                            MorphAnalysis(
                                surface=word,
                                stem=entry.stem or stem,
                                lemma=entry.lemma or stem,
                                pos=entry.pos,
                                flags=entry.flags,
                                suffix=rule.add,
                                inflectional=entry.inflectional,
                            )
                        )
                        if first_only:
                            return results

        # 3. Prefix-only stripping
        for flag, rules in self.prefix_rules.items():
            for rule in rules:
                stem = self._try_strip_prefix(word, rule)
                if stem is None or stem not in self.entries:
                    continue
                for entry in self.entries[stem]:
                    if flag in entry.flags:
                        results.append(
                            MorphAnalysis(
                                surface=word,
                                stem=entry.stem or stem,
                                lemma=entry.lemma or stem,
                                pos=entry.pos,
                                flags=entry.flags,
                                prefix=rule.add,
                                inflectional=entry.inflectional,
                            )
                        )
                        if first_only:
                            return results

        # 4. Prefix + suffix cross-product
        for pfx_flag, pfx_rules in self.prefix_rules.items():
            if not self.cross_product.get(pfx_flag, False):
                continue
            sfx_rules = self.suffix_rules.get(pfx_flag, [])
            if not sfx_rules:
                continue
            for pfx_rule in pfx_rules:
                after_pfx = self._try_strip_prefix(word, pfx_rule)
                if after_pfx is None:
                    continue
                for sfx_rule in sfx_rules:
                    stem = self._try_strip_suffix(after_pfx, sfx_rule)
                    if stem is None or stem not in self.entries:
                        continue
                    for entry in self.entries[stem]:
                        if pfx_flag in entry.flags:
                            results.append(
                                MorphAnalysis(
                                    surface=word,
                                    stem=entry.stem or stem,
                                    lemma=entry.lemma or stem,
                                    pos=entry.pos,
                                    flags=entry.flags,
                                    prefix=pfx_rule.add,
                                    suffix=sfx_rule.add,
                                    inflectional=entry.inflectional,
                                )
                            )
                            if first_only:
                                return results

        return results

    def _try_strip_suffix(
        self, word: str, rule: AffixRule
    ) -> Optional[str]:
        """Strip a suffix addition from a word, returning the
        reconstructed stem if the condition matches."""
        add = rule.add
        if add:
            if not word.endswith(add):
                return None
            base = word[: -len(add)]
        else:
            base = word

        if rule.strip:
            base = base + rule.strip

        if rule.condition.search(base):
            return base
        return None

    def _try_strip_prefix(
        self, word: str, rule: AffixRule
    ) -> Optional[str]:
        """Strip a prefix addition from a word, returning the
        reconstructed stem if the condition matches."""
        add = rule.add
        if add:
            if not word.startswith(add):
                return None
            base = word[len(add) :]
        else:
            base = word

        if rule.strip:
            base = rule.strip + base

        if rule.condition.search(base):
            return base
        return None

    # ── Form Generation ──────────────────────────────────────────

    def generate(self, stem: str, flag: str) -> list[str]:
        """Generate all valid inflected surface forms for a stem
        with a given flag.

        Applies prefix, suffix, and cross-product (prefix+suffix)
        rules, respecting morphophonological conditions.
        """
        forms: set[str] = {stem}

        # Suffix-only
        for rule in self.suffix_rules.get(flag, []):
            form = self._apply_suffix(stem, rule)
            if form:
                forms.add(form)

        # Prefix-only
        for rule in self.prefix_rules.get(flag, []):
            form = self._apply_prefix(stem, rule)
            if form:
                forms.add(form)

        # Cross-product: prefix + suffix
        if self.cross_product.get(flag, False):
            for pfx_rule in self.prefix_rules.get(flag, []):
                pfx_form = self._apply_prefix(stem, pfx_rule)
                if not pfx_form:
                    continue
                for sfx_rule in self.suffix_rules.get(flag, []):
                    form = self._apply_suffix(pfx_form, sfx_rule)
                    if form:
                        forms.add(form)

        return sorted(forms)

    def _apply_suffix(
        self, word: str, rule: AffixRule
    ) -> Optional[str]:
        """Apply a suffix rule to produce an inflected form."""
        if not rule.condition.search(word):
            return None
        base = word
        if rule.strip:
            if not base.endswith(rule.strip):
                return None
            base = base[: -len(rule.strip)]
        return base + rule.add if rule.add else base

    def _apply_prefix(
        self, word: str, rule: AffixRule
    ) -> Optional[str]:
        """Apply a prefix rule to produce an inflected form."""
        if not rule.condition.search(word):
            return None
        base = word
        if rule.strip:
            if not base.startswith(rule.strip):
                return None
            base = base[len(rule.strip) :]
        return rule.add + base if rule.add else base

    # ── Lookup Helpers ───────────────────────────────────────────

    def lookup(self, word: str) -> list[LexiconEntry]:
        """Look up all dictionary entries for a surface form."""
        return self.entries.get(self.normalize(word), [])

    def get_pos(self, word: str) -> Optional[str]:
        """Get the primary POS tag for a dictionary word."""
        entries = self.lookup(word)
        return entries[0].pos if entries else None

    def is_transitive(self, entry: LexiconEntry) -> bool:
        """Check if a lexicon entry marks a transitive verb."""
        return any(f in self.TRANSITIVE_FLAGS for f in entry.flags)

    def is_intransitive(self, entry: LexiconEntry) -> bool:
        """Check if a lexicon entry marks an intransitive verb."""
        return any(f in self.INTRANSITIVE_FLAGS for f in entry.flags)

    def _is_verb_entry(self, entry: LexiconEntry) -> bool:
        """Check if an entry is a verb (by flag or POS tag)."""
        if any(f in self.VERB_FLAGS for f in entry.flags):
            return True
        return "verb" in entry.pos

    def verb_stems(
        self, transitivity: Optional[str] = None
    ) -> dict[str, list[LexiconEntry]]:
        """Return all verb stems, optionally filtered by transitivity.

        Args:
            transitivity: "trans", "intrans", or None (all verbs).
        """
        result: dict[str, list[LexiconEntry]] = {}
        for surface, entries in self.entries.items():
            for entry in entries:
                if not self._is_verb_entry(entry):
                    continue
                if transitivity == "trans" and not self.is_transitive(entry):
                    continue
                if transitivity == "intrans" and not self.is_intransitive(
                    entry
                ):
                    continue
                result.setdefault(surface, []).append(entry)
        return result

    def __repr__(self) -> str:
        return (
            f"SoraniLexicon(words={len(self.words)}, "
            f"pfx_rules={sum(len(v) for v in self.prefix_rules.values())}, "
            f"sfx_rules={sum(len(v) for v in self.suffix_rules.values())})"
        )
