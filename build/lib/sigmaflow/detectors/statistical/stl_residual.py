"""Seasonal-decomposition residual detector."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...core.base import BaseDetector
from ...core.signal_frame import SignalFrame

__all__ = ["STLResidualDetector", "seasonal_decompose", "detect_period"]


def detect_period(x: np.ndarray, max_period: int | None = None) -> int | None:
    """Estimate the dominant period (in samples) via autocorrelation.

    Returns None when no significant periodicity is found.
    """
    n = len(x)
    if n < 8:
        return None
    max_period = max_period or n // 2
    y = x - np.mean(x)
    denom = float(np.dot(y, y))
    if denom == 0:
        return None
    acf = np.correlate(y, y, mode="full")[n - 1 :] / denom
    # First local maximum of the ACF beyond lag 1 that is meaningfully positive.
    search = acf[: max_period + 1]
    best_lag, best_val = None, 0.3  # require correlation above 0.3
    for lag in range(2, len(search) - 1):
        if search[lag] > best_val and search[lag] >= search[lag - 1] and search[lag] >= search[lag + 1]:
            best_lag, best_val = lag, search[lag]
    return best_lag


def seasonal_decompose(x: np.ndarray, period: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Classical additive decomposition: x = trend + seasonal + residual.

    Trend is a centered moving average over one period; seasonal is the
    mean detrended value at each phase of the cycle.
    """
    window = period if period % 2 == 1 else period + 1
    trend = pd.Series(x).rolling(window, center=True, min_periods=1).mean().to_numpy()
    detrended = x - trend
    phases = np.arange(len(x)) % period
    seasonal_means = np.array([detrended[phases == p].mean() for p in range(period)])
    seasonal_means -= seasonal_means.mean()  # keep decomposition additive around trend
    seasonal = seasonal_means[phases]
    residual = detrended - seasonal
    return trend, seasonal, residual


class STLResidualDetector(BaseDetector):
    """Decompose the signal into trend + seasonal + residual, then z-score
    the residual.

    Raw threshold detectors alarm on every seasonal peak of a periodic
    signal (daily ocean temperature, diurnal grid load); removing the
    seasonal component first leaves only genuinely unexpected deviations.

    Uses a classical moving-average decomposition (``method='classical'``,
    the only method in this release; LOESS-based STL may come later as an
    optional dependency). ``period`` is in samples and is auto-detected
    from the autocorrelation function when not given.
    """

    name = "stl_residual"

    def __init__(
        self,
        period: int | None = None,
        residual_threshold: float = 3.0,
        method: str = "classical",
    ):
        super().__init__()
        if method != "classical":
            raise ValueError("only method='classical' is available in this release")
        self.period = period
        self.residual_threshold = float(residual_threshold)
        self.method = method

    def _auto_threshold(self, scores: np.ndarray) -> float:
        return self.residual_threshold

    def _score_channel(self, x: np.ndarray, channel: str, sf: SignalFrame) -> np.ndarray:
        clean = np.nan_to_num(x, nan=float(np.nanmean(x)))
        period = self.period or detect_period(clean)
        if period and 2 <= period <= len(clean) // 2:
            _, _, residual = seasonal_decompose(clean, int(period))
            edge = (int(period) + 1) // 2
        else:
            # No detectable seasonality: fall back to detrended residual.
            window = max(3, len(clean) // 10) | 1
            trend = pd.Series(clean).rolling(
                window, center=True, min_periods=1
            ).mean().to_numpy()
            residual = clean - trend
            edge = 0  # broad trend window: boundary bias is negligible
        # Robust z-score of the residual so anomalies don't inflate the scale.
        median = np.median(residual)
        mad = np.median(np.abs(residual - median))
        scale = 1.4826 * mad if mad > 0 else (np.std(residual) or 1.0)
        scores = np.abs(residual - median) / scale
        # The centered trend estimate is biased inside the first/last
        # half-window, producing spurious residuals there — mask them out.
        if edge and len(scores) > 2 * edge:
            scores[:edge] = 0.0
            scores[-edge:] = 0.0
        return scores
