"""Model architecture modules."""

from .baseline import BaselineGEC
from .morphology_aware import MorphologyAwareGEC

__all__ = [
    "BaselineGEC",
    "MorphologyAwareGEC",
]
