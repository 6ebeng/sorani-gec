"""
Agreement Module — Backward-Compatible Re-Export

This module re-exports everything from the three focused submodules so that
existing imports like ``from src.morphology.agreement import X`` keep working.

Internal structure:
  constants.py  — All linguistic constants and lexicons (F#1–F#256)
  graph.py      — AgreementEdge dataclass + AgreementGraph class
  builder.py    — Helper functions + build_agreement_graph()
"""

from .constants import *  # noqa: F401,F403
from .graph import AgreementEdge, AgreementGraph  # noqa: F401
from .builder import build_agreement_graph  # noqa: F401