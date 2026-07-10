"""Z-score detector."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...core.base import BaseDetector
from ...core.signal_frame import SignalFrame

__all__ = ["ZScoreDetector"]


class ZScoreDetector(BaseDetector):
    """Flag points whose z-score exceeds ``threshold``.

    With ``window_size`` set, mean and std are computed over a rolling
    window (handles slow nonstationarity); otherwise global statistics
    are used — learned in :meth:`fit`, or from the scored signal itself
    if the detector was never fitted.
    """

    name = "zscore"

    def __init__(self, window_size: int | None = None, threshold: float = 3.0):
        super().__init__()
        self.window_size = window_size
        self.threshold = float(threshold)
        self._stats: dict[str, tuple[float, float]] = {}

    def _auto_threshold(self, scores: np.ndarray) -> float:
        return self.threshold

    def _fit_channel(self, x: np.ndarray, channel: str, sf: SignalFrame) -> None:
        x = x[~np.isnan(x)]
        self._stats[channel] = (float(np.mean(x)), float(np.std(x)))

    def _score_channel(self, x: np.ndarray, channel: str, sf: SignalFrame) -> np.ndarray:
        if self.window_size:
            s = pd.Series(x)
            mean = s.rolling(self.window_size, min_periods=2).mean()
            std = s.rolling(self.window_size, min_periods=2).std()
            std = std.replace(0.0, np.nan)
            return np.abs((s - mean) / std).to_numpy()
        if channel in self._stats:
            mean, std = self._stats[channel]
        else:
            clean = x[~np.isnan(x)]
            mean, std = float(np.mean(clean)), float(np.std(clean))
        if std == 0:
            return np.zeros_like(x)
        return np.abs(x - mean) / std
