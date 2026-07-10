"""Modified z-score detector (median / MAD based)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...core.base import BaseDetector
from ...core.signal_frame import SignalFrame

__all__ = ["ModifiedZScoreDetector"]


class ModifiedZScoreDetector(BaseDetector):
    """Z-score using median and MAD — robust to outliers in the statistics.

    Score = 0.6745 * |x - median| / MAD (Iglewicz & Hoaglin). Default
    threshold 3.5 per their recommendation.
    """

    name = "modified_zscore"

    def __init__(self, window_size: int | None = None, threshold: float = 3.5):
        super().__init__()
        self.window_size = window_size
        self.threshold = float(threshold)
        self._stats: dict[str, tuple[float, float]] = {}

    def _auto_threshold(self, scores: np.ndarray) -> float:
        return self.threshold

    @staticmethod
    def _median_mad(x: np.ndarray) -> tuple[float, float]:
        median = float(np.median(x))
        mad = float(np.median(np.abs(x - median)))
        return median, mad

    def _fit_channel(self, x: np.ndarray, channel: str, sf: SignalFrame) -> None:
        self._stats[channel] = self._median_mad(x[~np.isnan(x)])

    def _score_channel(self, x: np.ndarray, channel: str, sf: SignalFrame) -> np.ndarray:
        if self.window_size:
            s = pd.Series(x)
            median = s.rolling(self.window_size, min_periods=2).median()
            mad = (s - median).abs().rolling(self.window_size, min_periods=2).median()
            mad = mad.replace(0.0, np.nan)
            return (0.6745 * (s - median).abs() / mad).to_numpy()
        if channel in self._stats:
            median, mad = self._stats[channel]
        else:
            median, mad = self._median_mad(x[~np.isnan(x)])
        if mad == 0:
            return np.zeros_like(x)
        return 0.6745 * np.abs(x - median) / mad
