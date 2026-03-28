"""
Data Augmentation for Sorani Kurdish GEC

Provides augmentation strategies to expand the synthetic training corpus:
- Synonym replacement using the morphological lexicon
- Random word swap and deletion (noise injection)

Note: Back-translation was considered as an augmentation strategy but is
not implemented. No publicly available Kurdish→Kurdish translation model
had sufficient quality at the time of development. The remaining three
strategies (synonym, swap, delete) provide adequate diversity.
"""

import logging
import random
from typing import Optional, TYPE_CHECKING

from .tokenize import sorani_word_tokenize

if TYPE_CHECKING:
    from ..morphology.lexicon import SoraniLexicon

logger = logging.getLogger(__name__)


class SoraniAugmenter:
    """Augmentation strategies for Sorani Kurdish sentence pairs."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def synonym_replace(
        self,
        sentence: str,
        lexicon: "SoraniLexicon",
        replace_prob: float = 0.1,
    ) -> str:
        """Replace words with synonyms from the lexicon.

        Args:
            sentence: Input sentence.
            lexicon: SoraniLexicon instance with suggestion capability.
            replace_prob: Probability of replacing each word.

        Returns:
            Augmented sentence with some words replaced.
        """
        words = sorani_word_tokenize(sentence)
        augmented = []
        for word in words:
            if self.rng.random() < replace_prob and lexicon.available:
                suggestions = lexicon.suggest(word)
                candidates = [s for s in suggestions if s != word]
                if candidates:
                    augmented.append(self.rng.choice(candidates))
                    continue
            augmented.append(word)
        return " ".join(augmented)

    def random_swap(self, sentence: str, n_swaps: int = 1) -> str:
        """Randomly swap adjacent words in the sentence.

        Args:
            sentence: Input sentence.
            n_swaps: Number of swap operations.

        Returns:
            Sentence with words swapped.
        """
        words = sorani_word_tokenize(sentence)
        if len(words) < 2:
            return sentence
        for _ in range(n_swaps):
            idx = self.rng.randint(0, len(words) - 2)
            words[idx], words[idx + 1] = words[idx + 1], words[idx]
        return " ".join(words)

    def random_deletion(self, sentence: str, delete_prob: float = 0.1) -> str:
        """Randomly delete words from the sentence.

        Args:
            sentence: Input sentence.
            delete_prob: Probability of deleting each word.

        Returns:
            Sentence with some words removed.
        """
        words = sorani_word_tokenize(sentence)
        if len(words) <= 1:
            return sentence
        remaining = [w for w in words if self.rng.random() >= delete_prob]
        return " ".join(remaining) if remaining else words[0]

    def augment_pair(
        self,
        source: str,
        target: str,
        strategy: str = "swap",
        lexicon: Optional["SoraniLexicon"] = None,
        **kwargs,
    ) -> tuple[str, str]:
        """Augment a (source, target) pair.

        Applies augmentation to both source and target consistently.
        For swap/deletion, only the source (corrupted) side is augmented.

        Args:
            source: Corrupted sentence.
            target: Clean sentence.
            strategy: One of "swap", "delete", "synonym".
            lexicon: Required for synonym strategy.

        Returns:
            (augmented_source, target) pair.
        """
        if strategy == "swap":
            return self.random_swap(source, **kwargs), target
        elif strategy == "delete":
            return self.random_deletion(source, **kwargs), target
        elif strategy == "synonym" and lexicon is not None:
            aug_target = self.synonym_replace(target, lexicon, **kwargs)
            aug_source = self.synonym_replace(source, lexicon, **kwargs)
            return aug_source, aug_target
        return source, target

    def augment_corpus(
        self,
        pairs: list[dict],
        strategies: list[str] = ["swap", "delete"],
        augment_ratio: float = 0.5,
        lexicon: Optional["SoraniLexicon"] = None,
    ) -> list[dict]:
        """Augment a corpus of sentence pairs.

        For each selected pair, generates one augmented version per strategy.

        Args:
            pairs: List of dicts with 'source' and 'target' keys.
            strategies: Augmentation strategies to apply.
            augment_ratio: Fraction of pairs to augment.
            lexicon: Required if 'synonym' is in strategies.

        Returns:
            Original pairs + augmented pairs.
        """
        augmented = list(pairs)
        indices = list(range(len(pairs)))
        self.rng.shuffle(indices)
        n_to_augment = int(len(pairs) * augment_ratio)

        for idx in indices[:n_to_augment]:
            pair = pairs[idx]
            for strategy in strategies:
                if strategy == "synonym" and lexicon is None:
                    continue
                aug_src, aug_tgt = self.augment_pair(
                    pair["source"], pair["target"],
                    strategy=strategy, lexicon=lexicon,
                )
                aug_pair = dict(pair)
                aug_pair["source"] = aug_src
                aug_pair["target"] = aug_tgt
                aug_pair["augmented"] = strategy
                augmented.append(aug_pair)

        logger.info(
            "Augmented corpus: %d original + %d augmented = %d total",
            len(pairs), len(augmented) - len(pairs), len(augmented),
        )
        return augmented
