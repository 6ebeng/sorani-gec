"""Evaluation modules for Sorani Kurdish GEC."""

from .f05_scorer import (
    GECMetrics,
    SpanEdit,
    compute_f05,
    evaluate_corpus,
    evaluate_corpus_by_type,
    evaluate_corpus_span,
    evaluate_corpus_with_sentences,
    evaluate_sentence,
    span_based_edits,
)
from .agreement_accuracy import AgreementChecker
from .m2_scorer import (
    M2Edit,
    M2Sentence,
    evaluate_m2,
    evaluate_m2_by_type,
    parse_m2_file,
    write_m2_file,
)
from .inter_rater import compute_inter_rater_agreement
from .gleu_scorer import compute_gleu, compute_gleu_per_sentence

__all__ = [
    "GECMetrics",
    "SpanEdit",
    "compute_f05",
    "evaluate_corpus",
    "evaluate_corpus_by_type",
    "evaluate_corpus_span",
    "evaluate_corpus_with_sentences",
    "evaluate_sentence",
    "span_based_edits",
    "AgreementChecker",
    "M2Edit",
    "M2Sentence",
    "evaluate_m2",
    "evaluate_m2_by_type",
    "parse_m2_file",
    "write_m2_file",
    "compute_inter_rater_agreement",
    "compute_gleu",
    "compute_gleu_per_sentence",
]
