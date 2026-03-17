import logging
from typing import List, Optional
import difflib

from ..morphology.lexicon import SoraniLexicon

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

    def correct_sentence(self, sentence: str) -> str:
        """Apply spell correction to each word in the sentence.

        Returns the sentence with misspelled words replaced by the
        best suggestion from the lexicon. Words already correct or
        without close matches are left unchanged.
        """
        if not self.is_available():
            return sentence
        words = sentence.split()
        corrected = []
        for word in words:
            suggestions = self.get_suggestions(word)
            corrected.append(suggestions[0] if suggestions else word)
        return " ".join(corrected)
