"""Configurable feature extraction for windowed detectors (spec §5.7).

Each timestep is represented by statistics of a centered window around
it. All features are computed vectorized over a sliding-window matrix,
so extraction stays fast on long signals.
"""

from __future__ import annotations

import numpy as np

__all__ = ["FeatureExtractor", "AVAILABLE_FEATURES", "DEFAULT_FEATURES"]

DEFAULT_FEATURES = ["value", "mean", "std", "min", "max", "slope"]

_EPS = 1e-12


def _windows(x: np.ndarray, window_size: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Sliding-window matrix with edge padding: one row per timestep."""
    w = max(3, int(window_size))
    if w % 2 == 0:
        w += 1
    half = w // 2
    padded = np.pad(x, half, mode="edge")
    return np.lib.stride_tricks.sliding_window_view(padded, w), x, w


def _slope(W, x, w, t, t_var):
    return (W @ t) / t_var


def _curvature(W, x, w, t, t_var):
    p2 = t * t - np.mean(t * t)
    return (W @ p2) / (float(p2 @ p2) + _EPS)


def _zero_crossing_rate(W, x, w, t, t_var):
    centered = W - W.mean(axis=1, keepdims=True)
    signs = np.sign(centered)
    signs[signs == 0] = 1
    return np.mean(np.diff(signs, axis=1) != 0, axis=1)


def _psd(W):
    spec = np.abs(np.fft.rfft(W - W.mean(axis=1, keepdims=True), axis=1)) ** 2
    return spec[:, 1:]  # drop DC


def _spectral_entropy(W, x, w, t, t_var):
    psd = _psd(W)
    p = psd / (psd.sum(axis=1, keepdims=True) + _EPS)
    ent = -(p * np.log(p + _EPS)).sum(axis=1)
    return ent / np.log(p.shape[1] + _EPS)


def _dominant_frequency(W, x, w, t, t_var):
    psd = _psd(W)
    return (np.argmax(psd, axis=1) + 1) / w  # cycles per sample


def _autocorr(W, lag):
    centered = W - W.mean(axis=1, keepdims=True)
    var = (centered * centered).sum(axis=1) + _EPS
    return (centered[:, :-lag] * centered[:, lag:]).sum(axis=1) / var


def _moment(W, order):
    centered = W - W.mean(axis=1, keepdims=True)
    std = W.std(axis=1) + _EPS
    return (centered**order).mean(axis=1) / std**order


AVAILABLE_FEATURES = {
    "value": lambda W, x, w, t, tv: x,
    "mean": lambda W, x, w, t, tv: W.mean(axis=1),
    "std": lambda W, x, w, t, tv: W.std(axis=1),
    "skewness": lambda W, x, w, t, tv: _moment(W, 3),
    "kurtosis": lambda W, x, w, t, tv: _moment(W, 4) - 3.0,
    "min": lambda W, x, w, t, tv: W.min(axis=1),
    "max": lambda W, x, w, t, tv: W.max(axis=1),
    "range": lambda W, x, w, t, tv: np.ptp(W, axis=1),
    "iqr": lambda W, x, w, t, tv: (np.percentile(W, 75, axis=1)
                                   - np.percentile(W, 25, axis=1)),
    "slope": _slope,
    "curvature": _curvature,
    "zero_crossing_rate": _zero_crossing_rate,
    "spectral_entropy": _spectral_entropy,
    "dominant_frequency": _dominant_frequency,
    "autocorrelation_lag1": lambda W, x, w, t, tv: _autocorr(W, 1),
    "autocorrelation_lag5": lambda W, x, w, t, tv: _autocorr(W, min(5, w - 1)),
    "energy": lambda W, x, w, t, tv: (W * W).sum(axis=1),
}


class FeatureExtractor:
    """Turn a 1-D signal into an (n_samples, n_features) matrix of
    windowed features.

    >>> extractor = FeatureExtractor(features=["mean", "std", "spectral_entropy"])
    >>> X = extractor.transform(x)
    """

    def __init__(self, features: list[str] | None = None, window_size: int = 25):
        features = list(features) if features else list(DEFAULT_FEATURES)
        unknown = [f for f in features if f not in AVAILABLE_FEATURES]
        if unknown:
            raise ValueError(
                f"unknown features {unknown}; available: {sorted(AVAILABLE_FEATURES)}"
            )
        self.features = features
        self.window_size = int(window_size)

    @property
    def feature_names(self) -> list[str]:
        return list(self.features)

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.nan_to_num(np.asarray(x, dtype=np.float64), nan=0.0)
        W, x, w = _windows(x, self.window_size)
        half = w // 2
        t = np.arange(w, dtype=np.float64) - half
        t_var = float(t @ t)
        cols = [
            np.nan_to_num(
                np.asarray(AVAILABLE_FEATURES[name](W, x, w, t, t_var),
                           dtype=np.float64),
                nan=0.0, posinf=0.0, neginf=0.0,
            )
            for name in self.features
        ]
        return np.column_stack(cols)

    def __repr__(self) -> str:
        return f"FeatureExtractor(features={self.features}, window_size={self.window_size})"
