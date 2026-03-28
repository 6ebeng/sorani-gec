import logging
import re
from typing import List, Optional
import difflib

from ..morphology.lexicon import SoraniLexicon
from .tokenize import sorani_word_tokenize

logger = logging.getLogger(__name__)

class SoraniSpellChecker:
    """Spell-checker for Sorani Kurdish using the standalone SoraniLexicon."""

    def __init__(self, dict_path: Optional[str] = None, max_suggestions: int = 3):
        self.max_suggestions = max_suggestions
        self._lexicon = SoraniLexicon(dic_path=dict_path)

    def is_available(self) -> bool:
        return self._lexicon.available

    def is_correct(self, word: str) -> bool:
        if not self.is_available():
            return True
        return self._lexicon.is_correct(word)

    def get_suggestions(self, word: str) -> List[str]:
        if not self.is_available() or self.is_correct(word):
            return [word]
        # Prefer linguistically-informed REP rule suggestions first
        rep_suggestions = self._lexicon.suggest(word)
        if rep_suggestions:
            return rep_suggestions[:self.max_suggestions]
        # Fall back to edit-distance matching against word list
        matches = difflib.get_close_matches(word, self._lexicon.words, n=self.max_suggestions, cutoff=0.7)
        return matches if matches else [word]

    # Regex to split trailing Kurdish/Arabic punctuation from a word
    _PUNCT_RE = re.compile(r'^(.+?)([.،؟!؛:»«\-]+)$')

    def correct_sentence(self, sentence: str, model_confidence: float = 0.0) -> str:
        """Apply spell correction to each word in the sentence.

        If *model_confidence* exceeds 0.9 the spell-checker defers to the
        model output, avoiding the risk of undoing high-confidence corrections.

        Returns the sentence with misspelled words replaced by the
        best suggestion from the lexicon. Words already correct or
        without close matches are left unchanged.
        """
        if not self.is_available():
            return sentence
        if model_confidence > 0.9:
            logger.debug("Skipping spell-check: model confidence %.2f > 0.9", model_confidence)
            return sentence
        words = sorani_word_tokenize(sentence)
        corrected = []
        for word in words:
            # Separate trailing punctuation so it doesn't interfere
            # with dictionary lookup.
            m = self._PUNCT_RE.match(word)
            if m:
                core, punct = m.group(1), m.group(2)
            else:
                core, punct = word, ""
            suggestions = self.get_suggestions(core)
            corrected.append((suggestions[0] if suggestions else core) + punct)
        return " ".join(corrected)
