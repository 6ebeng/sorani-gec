"""Error generation modules for synthetic corpus creation."""

from .base import BaseErrorGenerator
from .subject_verb import SubjectVerbErrorGenerator
from .noun_adjective import NounAdjectiveErrorGenerator
from .clitic import CliticErrorGenerator
from .tense_agreement import TenseAgreementErrorGenerator
from .syntax_roles import CaseRoleErrorGenerator
from .dialectal import DialectalParticipleErrorGenerator
from .relative_clause import RelativeClauseErrorGenerator
from .adversative import AdversativeConnectorErrorGenerator
from .participle_swap import ParticipleSwapErrorGenerator
from .orthography import OrthographicErrorGenerator
from .negative_concord import NegativeConcordErrorGenerator
from .vocative_imperative import VocativeImperativeErrorGenerator
from .conditional_agreement import ConditionalAgreementErrorGenerator
from .adverb_verb_tense import AdverbVerbTenseErrorGenerator
from .preposition_fusion import PrepositionFusionErrorGenerator
from .demonstrative_contraction import DemonstrativeContractionErrorGenerator
from .quantifier_agreement import QuantifierAgreementErrorGenerator
from .possessive_clitic import PossessiveCliticErrorGenerator
from .polite_imperative import PoliteImperativeErrorGenerator
from .pipeline import ErrorPipeline

__all__ = [
    "BaseErrorGenerator",
    "SubjectVerbErrorGenerator",
    "NounAdjectiveErrorGenerator",
    "CliticErrorGenerator",
    "TenseAgreementErrorGenerator",
    "CaseRoleErrorGenerator",
    "DialectalParticipleErrorGenerator",
    "RelativeClauseErrorGenerator",
    "AdversativeConnectorErrorGenerator",
    "ParticipleSwapErrorGenerator",
    "OrthographicErrorGenerator",
    "NegativeConcordErrorGenerator",
    "VocativeImperativeErrorGenerator",
    "ConditionalAgreementErrorGenerator",
    "AdverbVerbTenseErrorGenerator",
    "PrepositionFusionErrorGenerator",
    "DemonstrativeContractionErrorGenerator",
    "QuantifierAgreementErrorGenerator",
    "PossessiveCliticErrorGenerator",
    "PoliteImperativeErrorGenerator",
    "ErrorPipeline",
]
