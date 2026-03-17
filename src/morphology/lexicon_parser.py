"""Backward-compatibility shim — imports from lexicon.py.

New code should use ``from src.morphology.lexicon import SoraniLexicon``
directly. This module keeps the old ``AhmadiLexiconParser`` name
available for any remaining call-sites.
"""

from .lexicon import SoraniLexicon as AhmadiLexiconParser  # noqa: F401
from .lexicon import LexiconEntry  # noqa: F401
