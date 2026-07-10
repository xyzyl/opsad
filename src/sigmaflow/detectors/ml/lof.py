"""Local Outlier Factor detector on windowed features."""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import LocalOutlierFactor

from ...core.base import BaseDetector
from ...core.signal_frame import SignalFrame
from ._features import window_features

__all__ = ["LOFDetector"]


class LOFDetector(BaseDetector):
    """Local Outlier Factor over per-timestep window features.

    Density-based: a timestep is anomalous when its feature-space
    neighborhood is sparser than its neighbors' — catching values that
    are locally unusual even when globally in range. Scores are the
    negated sklearn decision function (auto threshold 0, matching
    sklearn's own contamination-based labeling).
    """

    name = "lof"

    def __init__(
        self,
        n_neighbors: int = 20,
        contamination="auto",
        window_size: int = 25,
    ):
        super().__init__()
        self.n_neighbors = int(n_neighbors)
        self.contamination = contamination
        self.window_size = int(window_size)
        self._models: dict[str, LocalOutlierFactor] = {}

    def _auto_threshold(self, scores: np.ndarray) -> float:
        return 0.0

    def _fit_channel(self, x: np.ndarray, channel: str, sf: SignalFrame) -> None:
        feats = window_features(x, self.window_size)
        n_neighbors = min(self.n_neighbors, len(feats) - 1)
        model = LocalOutlierFactor(
            n_neighbors=max(1, n_neighbors),
            contamination=self.contamination,
            novelty=True,
        )
        model.fit(feats)
        self._models[channel] = model

    def _score_channel(self, x: np.ndarray, channel: str, sf: SignalFrame) -> np.ndarray:
        if channel not in self._models:
            self._fit_channel(x, channel, sf)
        feats = window_features(x, self.window_size)
        return -self._models[channel].decision_function(feats)
