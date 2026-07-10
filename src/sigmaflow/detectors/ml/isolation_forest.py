"""Isolation Forest detector on windowed features."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest

from ...core.base import BaseDetector
from ...core.signal_frame import SignalFrame
from ._features import window_features

__all__ = ["IsolationForestDetector"]


class IsolationForestDetector(BaseDetector):
    """Isolation Forest over per-timestep window features.

    Each timestep becomes a feature vector (window mean/std/min/max/slope
    plus the raw value); anomalies are isolated in fewer random splits.
    Scores are the negated sklearn decision function, so scores above 0
    are what sklearn itself would label anomalous at the configured
    ``contamination`` — which is why the auto threshold is 0.
    """

    name = "isolation_forest"

    def __init__(
        self,
        n_estimators: int = 200,
        contamination="auto",
        window_size: int = 25,
        random_state: int | None = 0,
    ):
        super().__init__()
        self.n_estimators = int(n_estimators)
        self.contamination = contamination
        self.window_size = int(window_size)
        self.random_state = random_state
        self._models: dict[str, IsolationForest] = {}

    def _auto_threshold(self, scores: np.ndarray) -> float:
        return 0.0

    def _fit_channel(self, x: np.ndarray, channel: str, sf: SignalFrame) -> None:
        model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
        )
        model.fit(window_features(x, self.window_size))
        self._models[channel] = model

    def _score_channel(self, x: np.ndarray, channel: str, sf: SignalFrame) -> np.ndarray:
        if channel not in self._models:
            self._fit_channel(x, channel, sf)
        feats = window_features(x, self.window_size)
        return -self._models[channel].decision_function(feats)
