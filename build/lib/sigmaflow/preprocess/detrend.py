"""Detrend: remove trend components from a signal."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.base import BasePreprocessor
from ..core.signal_frame import SignalFrame, time_to_seconds

__all__ = ["Detrend"]

_METHODS = ("linear", "polynomial", "moving_average", "differencing")


class Detrend(BasePreprocessor):
    """Remove the trend component of each channel.

    Methods:
      - ``linear``: subtract the least-squares line.
      - ``polynomial``: subtract a fitted polynomial of ``order``.
      - ``moving_average``: subtract a centered rolling mean of ``window`` samples.
      - ``differencing``: first difference (output[0] = 0).

    The trend removed by the most recent :meth:`transform` is stored, so
    :meth:`inverse_transform` can restore a signal of the same length.
    """

    name = "detrend"

    def __init__(self, method: str = "linear", order: int = 2, window: int = 101):
        if method not in _METHODS:
            raise ValueError(f"unknown detrend method {method!r}; choose from {_METHODS}")
        self.method = method
        self.order = int(order)
        self.window = int(window)
        self._trends: dict[str, np.ndarray] = {}
        self._x0: dict[str, float] = {}

    def transform(self, sf: SignalFrame) -> SignalFrame:
        secs = time_to_seconds(sf.time)
        t = secs - secs[0]
        data = {}
        self._trends, self._x0 = {}, {}
        for c in sf.channels:
            x = sf[c].to_numpy()
            if self.method == "differencing":
                out = np.diff(x, prepend=x[0])
                self._x0[c] = float(x[0])
            else:
                trend = self._estimate_trend(t, x)
                self._trends[c] = trend
                out = x - trend
            data[c] = out
        df = pd.DataFrame(data, index=sf.time)
        return sf._with(df)

    def inverse_transform(self, sf: SignalFrame) -> SignalFrame:
        data = {}
        for c in sf.channels:
            x = sf[c].to_numpy()
            if self.method == "differencing":
                if c not in self._x0:
                    raise RuntimeError("inverse_transform requires a prior transform")
                restored = np.cumsum(x)
                restored += self._x0[c] - restored[0]
                data[c] = restored
            else:
                trend = self._trends.get(c)
                if trend is None or len(trend) != len(x):
                    raise RuntimeError(
                        "inverse_transform requires a prior transform of the same length"
                    )
                data[c] = x + trend
        df = pd.DataFrame(data, index=sf.time)
        return sf._with(df)

    def _estimate_trend(self, t: np.ndarray, x: np.ndarray) -> np.ndarray:
        if self.method == "linear":
            coeffs = np.polyfit(t, x, 1)
            return np.polyval(coeffs, t)
        if self.method == "polynomial":
            coeffs = np.polyfit(t, x, self.order)
            return np.polyval(coeffs, t)
        # moving_average
        window = min(self.window, len(x))
        if window % 2 == 0:
            window = max(1, window - 1)
        return (
            pd.Series(x)
            .rolling(window, center=True, min_periods=1)
            .mean()
            .to_numpy()
        )
