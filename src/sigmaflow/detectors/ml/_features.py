"""Minimal windowed feature extraction for ML detectors.

Each timestep is represented by statistics of a centered window around
it: mean, std, min, max, slope, and the raw center value. The full
configurable ``sigmaflow.features`` module (spec §5.7) replaces this in
a later release; ML detectors take the feature matrix through a single
function so the swap is one call site.
"""

from __future__ import annotations

import numpy as np

__all__ = ["window_features", "FEATURE_NAMES"]

FEATURE_NAMES = ["value", "mean", "std", "min", "max", "slope"]


def window_features(x: np.ndarray, window_size: int) -> np.ndarray:
    """Return an (n_samples, 6) feature matrix for a 1-D signal.

    Edges are handled by padding with edge values, so the output always
    has one row per input timestep.
    """
    x = np.nan_to_num(np.asarray(x, dtype=np.float64), nan=0.0)
    w = max(3, int(window_size))
    if w % 2 == 0:
        w += 1
    half = w // 2
    padded = np.pad(x, half, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, w)

    t = np.arange(w) - half
    t_var = float(np.dot(t, t))
    slope = (windows @ t) / t_var  # least-squares slope of each window

    feats = np.column_stack([
        x,
        windows.mean(axis=1),
        windows.std(axis=1),
        windows.min(axis=1),
        windows.max(axis=1),
        slope,
    ])
    return feats
