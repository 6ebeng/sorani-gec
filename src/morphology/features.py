"""
Morphological Feature Extraction

Extracts structured feature bundles (person, number, tense, aspect,
case, definiteness, transitivity, clitic_person, clitic_number) from
morphological analysis output for use in model embeddings.
"""

import logging
from typing import Optional
from .analyzer import MorphologicalAnalyzer, MorphFeatures

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extract morphological feature vectors for model input."""
    
    def __init__(self, analyzer: Optional[MorphologicalAnalyzer] = None):
        self.analyzer = analyzer or MorphologicalAnalyzer()
        self.feature_vocab = self.analyzer.build_feature_vocabulary()
    
    def extract_features(self, sentence: str) -> list[list[int]]:
        """Extract feature vectors for each token in a sentence.
        
        Returns:
            List of feature index vectors, one per token.
            Each vector has 9 elements: [person, number, tense, aspect, case, definiteness, transitivity, clitic_person, clitic_number]
        """
        morph_features = self.analyzer.analyze_sentence(sentence)
        return [feat.to_vector_indices(self.feature_vocab) for feat in morph_features]

    def get_vocab_size(self) -> int:
        """Return the total feature vocabulary size."""
        return len(self.feature_vocab)

    def get_num_features(self) -> int:
        """Return number of feature types."""
        return 9
