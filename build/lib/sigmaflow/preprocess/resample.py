"""Resampler: change the sample rate of a signal."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.base import BasePreprocessor
from ..core.signal_frame import SignalFrame, time_to_seconds

__all__ = ["Resampler"]

_DOWN_METHODS = ("mean", "median", "max", "min")
_UP_METHODS = ("linear", "cubic")


class Resampler(BasePreprocessor):
    """Resample a signal to ``target_rate`` Hz onto a uniform grid.

    Downsampling aggregates samples into bins (``mean``, ``median``,
    ``max``, ``min``), optionally low-pass filtering first to prevent
    aliasing. Upsampling interpolates (``linear``, ``cubic``). If
    ``method`` is None a sensible one is picked per direction.
    """

    name = "resampler"

    def __init__(
        self,
        target_rate: float,
        method: str | None = None,
        anti_alias: bool = True,
    ):
        if target_rate <= 0:
            raise ValueError("target_rate must be positive")
        if method is not None and method not in _DOWN_METHODS + _UP_METHODS:
            raise ValueError(f"unknown resampling method {method!r}")
        self.target_rate = float(target_rate)
        self.method = method
        self.anti_alias = anti_alias

    def transform(self, sf: SignalFrame) -> SignalFrame:
        secs = time_to_seconds(sf.time)
        if len(secs) < 2:
            return sf
        current_rate = sf.sample_rate or 1.0 / float(np.median(np.diff(secs)))
        downsampling = self.target_rate < current_rate

        method = self.method or ("mean" if downsampling else "linear")
        if downsampling and method in _UP_METHODS:
            raise ValueError(f"method {method!r} is an interpolation method; "
                             "downsampling needs mean/median/max/min")
        if not downsampling and method in _DOWN_METHODS:
            method = "linear"

        new_secs = np.arange(secs[0], secs[-1] + 1e-12, 1.0 / self.target_rate)
        data = {}
        for c in sf.channels:
            x = sf[c].to_numpy()
            if downsampling:
                y = x
                if self.anti_alias and method == "mean":
                    y = self._lowpass(x, current_rate)
                data[c] = self._bin_aggregate(secs, y, new_secs, method)
            else:
                data[c] = self._interpolate(secs, x, new_secs, method)

        if isinstance(sf.time, pd.DatetimeIndex):
            index = pd.DatetimeIndex((new_secs * 1e9).astype("int64").view("datetime64[ns]"))
        else:
            index = pd.Index(new_secs)
        df = pd.DataFrame(data, index=index)
        return sf._with(df, sample_rate=self.target_rate, anomaly_labels=None)

    # ---------------------------------------------------------------- #

    def _lowpass(self, x: np.ndarray, current_rate: float) -> np.ndarray:
        from scipy.signal import butter, filtfilt

        nyquist = current_rate / 2.0
        cutoff = 0.45 * self.target_rate
        if cutoff >= nyquist or len(x) < 15:
            return x
        b, a = butter(4, cutoff / nyquist)
        return filtfilt(b, a, x)

    @staticmethod
    def _bin_aggregate(
        secs: np.ndarray, x: np.ndarray, new_secs: np.ndarray, method: str
    ) -> np.ndarray:
        # Each output point aggregates the input samples falling into its bin.
        bin_width = new_secs[1] - new_secs[0] if len(new_secs) > 1 else np.inf
        edges = np.append(new_secs, new_secs[-1] + bin_width)
        idx = np.searchsorted(edges, secs, side="right") - 1
        idx = np.clip(idx, 0, len(new_secs) - 1)
        agg = {"mean": np.mean, "median": np.median, "max": np.max, "min": np.min}[method]
        out = np.empty(len(new_secs))
        for i in range(len(new_secs)):
            members = x[idx == i]
            out[i] = agg(members) if len(members) else np.nan
        # Fill any empty bins by interpolation so output stays gap-free.
        if np.isnan(out).any():
            valid = ~np.isnan(out)
            out = np.interp(new_secs, new_secs[valid], out[valid])
        return out

    @staticmethod
    def _interpolate(
        secs: np.ndarray, x: np.ndarray, new_secs: np.ndarray, method: str
    ) -> np.ndarray:
        if method == "cubic" and len(x) >= 4:
            from scipy.interpolate import CubicSpline

            return CubicSpline(secs, x)(new_secs)
        return np.interp(new_secs, secs, x)
