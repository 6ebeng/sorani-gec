"""
Ensemble GEC Model

Combines multiple GEC models (baseline + morphology-aware) for improved
correction quality via majority voting or score-based selection.
"""

import logging
from collections import Counter
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class EnsembleGEC(nn.Module):
    """Ensemble that combines outputs from multiple GEC models.

    Supports two combination strategies:
    - majority_vote: Pick the most common correction across models.
    - best_score: Pick the correction with highest average log-probability
      (requires models that return scores).
    """

    def __init__(
        self,
        models: list[nn.Module],
        strategy: str = "majority_vote",
    ):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.strategy = strategy

    def correct(self, text: str, num_beams: int = 4, **kwargs) -> str:
        """Correct a single sentence using the ensemble."""
        candidates: list[str] = []
        scores: list[float] = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                if hasattr(model, "correct_with_morphology"):
                    analyzer = kwargs.get("analyzer")
                    feature_extractor = kwargs.get("feature_extractor")
                    if analyzer and feature_extractor:
                        if self.strategy == "best_score" and hasattr(model, "correct_with_confidence"):
                            corrected, conf = model.correct_with_confidence(
                                text, analyzer, feature_extractor, num_beams=num_beams
                            )
                            scores.append(conf)
                        else:
                            corrected = model.correct_with_morphology(  # type: ignore[operator]
                                text, analyzer, feature_extractor, num_beams=num_beams
                            )
                            scores.append(0.0)
                    else:
                        logger.warning(
                            "MorphologyAwareGEC falling back to baseline: "
                            "missing feature_extractor"
                        )
                        corrected = model.correct(text, num_beams=num_beams)  # type: ignore[operator]
                        scores.append(0.0)
                else:
                    if self.strategy == "best_score" and hasattr(model, "correct_with_confidence"):
                        corrected, conf = model.correct_with_confidence(text, num_beams=num_beams)
                        scores.append(conf)
                    else:
                        corrected = model.correct(text, num_beams=num_beams)  # type: ignore[operator]
                        scores.append(0.0)
                candidates.append(corrected)

        if self.strategy == "majority_vote":
            return self._majority_vote(candidates, text)
        if self.strategy == "best_score":
            if scores and any(s > 0 for s in scores):
                best_idx = max(range(len(scores)), key=lambda i: scores[i])
                return candidates[best_idx]
            return candidates[0] if candidates else text
        raise ValueError(f"Unknown ensemble strategy: {self.strategy!r}")

    def correct_batch(self, texts: list[str], num_beams: int = 4, **kwargs) -> list[str]:
        """Correct a batch of sentences using the ensemble.

        Delegates to each model's ``correct_batch`` when available for
        better throughput (batched inference), then applies the ensemble
        strategy per-sentence across model outputs.
        """
        if not texts:
            return []
        # Collect per-model batch outputs
        all_outputs: list[list[str]] = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                if hasattr(model, "correct_batch"):
                    batch_out = model.correct_batch(texts, num_beams=num_beams, **kwargs)
                else:
                    batch_out = [model.correct(t, num_beams=num_beams) for t in texts]
                all_outputs.append(batch_out)

        # Apply ensemble strategy per-sentence
        results: list[str] = []
        for i, text in enumerate(texts):
            candidates = [out[i] for out in all_outputs if i < len(out)]
            if self.strategy == "majority_vote":
                results.append(self._majority_vote(candidates, text))
            else:
                results.append(candidates[0] if candidates else text)
        return results

    @staticmethod
    def _majority_vote(candidates: list[str], fallback: str) -> str:
        """Return the most common candidate; fallback to first if all differ."""
        if not candidates:
            return fallback
        counts = Counter(candidates)
        winner, count = counts.most_common(1)[0]
        return winner
