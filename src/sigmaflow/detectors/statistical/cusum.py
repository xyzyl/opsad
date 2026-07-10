"""CUSUM detector: cumulative sum control chart for mean-shift detection."""

from __future__ import annotations

import numpy as np

from ...core.base import BaseDetector
from ...core.signal_frame import SignalFrame

__all__ = ["CUSUMDetector"]


class CUSUMDetector(BaseDetector):
    """Detect persistent shifts in the signal mean (two-sided CUSUM).

    Maintains upper and lower cumulative sums of standardized deviations
    from ``target``; the sums accumulate only when deviations exceed
    ``drift`` (in units of the signal's standard deviation). The score is
    the larger of the two sums, so a sustained shift builds up score even
    when each individual point looks normal. Suited to regime changes,
    sensor drift, and gradual degradation.

    Parameters
    ----------
    target : expected mean; estimated from fit data (or the scored signal)
        if not provided.
    threshold : alarm level in std units (classic choice: 4–5).
    drift : allowable slack in std units before accumulation (typically
        half the shift size you want to detect).
    """

    name = "cusum"

    def __init__(
        self,
        target: float | None = None,
        threshold: float = 5.0,
        drift: float = 0.5,
    ):
        super().__init__()
        self.target = target
        self.threshold = float(threshold)
        self.drift = float(drift)
        self._stats: dict[str, tuple[float, float]] = {}

    def _auto_threshold(self, scores: np.ndarray) -> float:
        return self.threshold

    def _fit_channel(self, x: np.ndarray, channel: str, sf: SignalFrame) -> None:
        x = x[~np.isnan(x)]
        target = self.target if self.target is not None else float(np.mean(x))
        std = float(np.std(x))
        self._stats[channel] = (target, std if std > 0 else 1.0)

    def _score_channel(self, x: np.ndarray, channel: str, sf: SignalFrame) -> np.ndarray:
        if channel in self._stats:
            target, std = self._stats[channel]
        else:
            clean = x[~np.isnan(x)]
            target = self.target if self.target is not None else float(np.median(clean))
            std = float(np.std(clean)) or 1.0

        z = (np.nan_to_num(x, nan=target) - target) / std
        s_hi = np.empty(len(x))
        s_lo = np.empty(len(x))
        hi = lo = 0.0
        for i, zi in enumerate(z):
            hi = max(0.0, hi + zi - self.drift)
            lo = max(0.0, lo - zi - self.drift)
            s_hi[i] = hi
            s_lo[i] = lo
        return np.maximum(s_hi, s_lo)
