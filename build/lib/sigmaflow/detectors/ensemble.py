"""Ensemble detector: combine multiple detectors' judgments."""

from __future__ import annotations

import numpy as np

from ..core.base import BaseDetector
from ..core.signal_frame import SignalFrame

__all__ = ["EnsembleDetector"]

_AGGREGATIONS = ("mean", "max", "voting", "weighted")


class EnsembleDetector(BaseDetector):
    """Aggregate several detectors into one.

    Different detectors excel at different anomaly types (Z-score at
    point spikes, CUSUM at drift, Isolation Forest at odd shapes);
    ensembling catches them all at once.

    Member scores live on wildly different scales (a |z| of 4 vs. a
    CUSUM sum of 900), so each member's scores are robustly standardized
    (median/MAD) before aggregation:

    - ``mean`` / ``max``: combine standardized scores.
    - ``weighted``: weighted mean with user ``weights``.
    - ``voting``: each member votes with its own binary labels (its own
      auto threshold); the score is the fraction of alarming members and
      the auto threshold is a strict majority.
    """

    name = "ensemble"

    def __init__(self, detectors: list[BaseDetector], aggregation: str = "mean",
                 weights: list[float] | None = None):
        super().__init__()
        if not detectors:
            raise ValueError("EnsembleDetector needs at least one member detector")
        if aggregation not in _AGGREGATIONS:
            raise ValueError(
                f"unknown aggregation {aggregation!r}; choose from {_AGGREGATIONS}"
            )
        if aggregation == "weighted":
            if weights is None or len(weights) != len(detectors):
                raise ValueError("weighted aggregation needs one weight per detector")
        self.detectors = list(detectors)
        self.aggregation = aggregation
        self.weights = list(weights) if weights else None

    def _score_channel(self, x, channel, sf):  # pragma: no cover - not used
        raise NotImplementedError("EnsembleDetector overrides score() directly")

    @staticmethod
    def _standardize(scores: np.ndarray) -> np.ndarray:
        median = np.median(scores)
        mad = np.median(np.abs(scores - median))
        scale = 1.4826 * mad if mad > 0 else (np.std(scores) or 1.0)
        return (scores - median) / scale

    def fit(self, sf: SignalFrame) -> "EnsembleDetector":
        for det in self.detectors:
            det.fit(sf)
        self._fitted = True
        return self

    def score(self, sf: SignalFrame) -> np.ndarray:
        member_scores = [det.score(sf) for det in self.detectors]
        if self.aggregation == "voting":
            votes = []
            for det, scores in zip(self.detectors, member_scores):
                thr = det._resolve_threshold(scores)
                votes.append((scores > thr).astype(float))
            return np.mean(np.vstack(votes), axis=0)
        standardized = np.vstack([self._standardize(s) for s in member_scores])
        if self.aggregation == "max":
            return np.max(standardized, axis=0)
        if self.aggregation == "weighted":
            w = np.asarray(self.weights, dtype=float)
            return (w[:, None] * standardized).sum(axis=0) / w.sum()
        return np.mean(standardized, axis=0)

    def _auto_threshold(self, scores: np.ndarray) -> float:
        if self.aggregation == "voting":
            return 0.5  # strict majority
        return float(np.mean(scores) + 3.0 * np.std(scores))

    def get_params(self) -> dict:
        return {
            "detectors": [type(d).__name__ for d in self.detectors],
            "aggregation": self.aggregation,
            "weights": self.weights,
        }
