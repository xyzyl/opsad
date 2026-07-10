"""Normalizer: scale channel values."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.base import BasePreprocessor
from ..core.signal_frame import SignalFrame

__all__ = ["Normalizer"]

_METHODS = ("z_score", "min_max", "robust")


class Normalizer(BasePreprocessor):
    """Scale each channel.

    Methods:
      - ``z_score``: (x - mean) / std.
      - ``min_max``: (x - min) / (max - min), mapped to [0, 1].
      - ``robust``: (x - median) / IQR — resistant to outliers.

    ``fit_on`` restricts statistic estimation to a (start, end) time
    interval known to be normal, so anomalies don't contaminate the
    scaling. Fully invertible via :meth:`inverse_transform`.
    """

    name = "normalizer"

    def __init__(self, method: str = "z_score", fit_on: tuple | None = None):
        if method not in _METHODS:
            raise ValueError(f"unknown normalization method {method!r}; choose from {_METHODS}")
        self.method = method
        self.fit_on = tuple(fit_on) if fit_on else None
        self._center: dict[str, float] = {}
        self._scale: dict[str, float] = {}

    def fit(self, sf: SignalFrame) -> "Normalizer":
        source = sf.slice(*self.fit_on) if self.fit_on else sf
        self._center, self._scale = {}, {}
        for c in source.channels:
            x = source[c].to_numpy()
            x = x[~np.isnan(x)]
            if len(x) == 0:
                center, scale = 0.0, 1.0
            elif self.method == "z_score":
                center, scale = float(np.mean(x)), float(np.std(x))
            elif self.method == "min_max":
                center, scale = float(np.min(x)), float(np.max(x) - np.min(x))
            else:  # robust
                q1, q2, q3 = np.percentile(x, [25, 50, 75])
                center, scale = float(q2), float(q3 - q1)
            self._center[c] = center
            self._scale[c] = scale if scale > 0 else 1.0
        return self

    def transform(self, sf: SignalFrame) -> SignalFrame:
        if not self._center:
            self.fit(sf)
        data = {}
        for c in sf.channels:
            if c not in self._center:
                raise KeyError(f"Normalizer was not fitted on channel {c!r}")
            data[c] = (sf[c].to_numpy() - self._center[c]) / self._scale[c]
        df = pd.DataFrame(data, index=sf.time)
        return sf._with(df)

    def inverse_transform(self, sf: SignalFrame) -> SignalFrame:
        if not self._center:
            raise RuntimeError("inverse_transform requires a prior fit")
        data = {}
        for c in sf.channels:
            data[c] = sf[c].to_numpy() * self._scale[c] + self._center[c]
        df = pd.DataFrame(data, index=sf.time)
        return sf._with(df)
