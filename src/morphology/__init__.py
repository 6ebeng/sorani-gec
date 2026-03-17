"""Morphological analysis modules for Sorani Kurdish."""

from .analyzer import MorphologicalAnalyzer, MorphFeatures
from .features import FeatureExtractor
from .graph import AgreementEdge, AgreementGraph
from .builder import build_agreement_graph
from .lexicon import SoraniLexicon

__all__ = [
    "MorphologicalAnalyzer",
    "MorphFeatures",
    "FeatureExtractor",
    "AgreementEdge",
    "AgreementGraph",
    "build_agreement_graph",
    "SoraniLexicon",
]
