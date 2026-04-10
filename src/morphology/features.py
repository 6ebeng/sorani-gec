"""
Morphological Feature Extraction

Extracts structured feature bundles (person, number, tense, aspect,
case, definiteness, transitivity, clitic_person, clitic_number) from
morphological analysis output for use in model embeddings.

The POS_CATEGORIES and SORANI_GENDER_CATEGORIES constants from
constants.py (F#263, F#266; Haji Marf) define the full taxonomy of
POS tags and gender categories for Sorani Kurdish; this module
extracts a subset of those features into numeric vectors.
"""

import logging
from typing import Optional
from .analyzer import MorphologicalAnalyzer, MorphFeatures
from .constants import POS_CATEGORIES, SORANI_GENDER_CATEGORIES

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extract morphological feature vectors for model input."""
    
    _NUM_FEATURES = 9

    def __init__(self, analyzer: Optional[MorphologicalAnalyzer] = None):
        self.analyzer = analyzer or MorphologicalAnalyzer()
        self.feature_vocab = self.analyzer.build_feature_vocabulary()
        # Validate feature vector length matches expected count (PIPE-8)
        test_feats = self.analyzer.analyze_token("تۆ")
        if test_feats is not None:
            vec = test_feats.to_vector_indices(self.feature_vocab)
            assert len(vec) == self._NUM_FEATURES, (
                "Feature vector length mismatch: analyzer produces %d features "
                "but FeatureExtractor expects %d" % (len(vec), self._NUM_FEATURES)
            )
    
    def extract_features(self, sentence: str) -> list[list[int]]:
        """Extract feature vectors for each token in a sentence.
        
        Returns:
            List of feature index vectors, one per token.
            Each vector has 9 elements: [person, number, tense, aspect, case, definiteness, transitivity, clitic_person, clitic_number]
        """
        try:
            morph_features = self.analyzer.analyze_sentence(sentence)
        except Exception:
            logger.warning("analyze_sentence() failed for: %.50s", sentence)
            return []
        return [feat.to_vector_indices(self.feature_vocab) for feat in morph_features]

    def get_vocab_size(self) -> int:
        """Return the total feature vocabulary size."""
        return len(self.feature_vocab)

    def get_num_features(self) -> int:
        """Return number of feature types."""
        return self._NUM_FEATURES
