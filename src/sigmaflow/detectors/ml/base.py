"""Shared machinery for windowed feature-space ML detectors."""

from __future__ import annotations

import numpy as np

from ...core.base import BaseDetector
from ...core.signal_frame import SignalFrame
from ...features.extractors import FeatureExtractor

__all__ = ["WindowedMLDetector"]


class WindowedMLDetector(BaseDetector):
    """Base for detectors that fit an sklearn outlier model on windowed
    features.

    Two channel modes:

    - default: one model per channel, scores aggregated with the
      pointwise max (a timestep is as anomalous as its worst channel);
    - ``multivariate=True``: one joint model over all channels' features
      stacked side by side — catches anomalies visible only in the
      *relationship* between channels.
    """

    def __init__(self, window_size: int = 25, features: list[str] | None = None,
                 multivariate: bool = False):
        super().__init__()
        self.window_size = int(window_size)
        self.features = list(features) if features else None
        self.multivariate = bool(multivariate)
        self._extractor = FeatureExtractor(features=self.features,
                                           window_size=self.window_size)
        self._models: dict[str, object] = {}

    def _auto_threshold(self, scores: np.ndarray) -> float:
        # scores are the negated sklearn decision function: >0 is what
        # sklearn itself would label anomalous at the set contamination
        return 0.0

    def _make_model(self):  # pragma: no cover - abstract
        raise NotImplementedError

    # ------------------------------------------------------------ per-channel
    def _fit_channel(self, x: np.ndarray, channel: str, sf: SignalFrame) -> None:
        model = self._make_model()
        model.fit(self._extractor.transform(x))
        self._models[channel] = model

    def _score_channel(self, x: np.ndarray, channel: str, sf: SignalFrame) -> np.ndarray:
        if channel not in self._models:
            self._fit_channel(x, channel, sf)
        return -self._models[channel].decision_function(self._extractor.transform(x))

    # ------------------------------------------------------------ multivariate
    def _stacked_features(self, sf: SignalFrame) -> np.ndarray:
        return np.hstack([
            self._extractor.transform(sf[c].to_numpy()) for c in sf.channels
        ])

    def fit(self, sf: SignalFrame):
        if not self.multivariate:
            return super().fit(sf)
        model = self._make_model()
        model.fit(self._stacked_features(sf))
        self._models["__joint__"] = model
        self._fitted = True
        return self

    def score(self, sf: SignalFrame) -> np.ndarray:
        if not self.multivariate:
            return super().score(sf)
        if "__joint__" not in self._models:
            self.fit(sf)
        scores = -self._models["__joint__"].decision_function(self._stacked_features(sf))
        return np.nan_to_num(np.asarray(scores, dtype=np.float64),
                             nan=0.0, posinf=0.0, neginf=0.0)
