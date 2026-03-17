"""Evaluation modules for Sorani Kurdish GEC."""

from .f05_scorer import GECMetrics, compute_f05, evaluate_corpus
from .agreement_accuracy import AgreementChecker

__all__ = [
    "GECMetrics",
    "compute_f05",
    "evaluate_corpus",
    "AgreementChecker",
]
