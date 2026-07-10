"""Threshold selection strategies for converting scores to labels."""

from __future__ import annotations

import numpy as np

__all__ = ["THRESHOLD_METHODS", "compute_threshold"]

THRESHOLD_METHODS = {"auto", "percentile", "sigma", "fixed"}


def compute_threshold(
    scores: np.ndarray, method: str, value: float | None = None
) -> float:
    """Compute a score threshold.

    - ``percentile``: top (100 - value)% of scores are anomalous (default 99).
    - ``sigma``: mean + value * std of the score distribution (default 3).
    - ``fixed``: use ``value`` directly.

    ``auto`` is resolved by each detector (see ``BaseDetector._auto_threshold``).
    """
    scores = np.asarray(scores, dtype=np.float64)
    if method == "percentile":
        p = 99.0 if value is None else float(value)
        if not 0 < p < 100:
            raise ValueError("percentile must be in (0, 100)")
        return float(np.percentile(scores, p))
    if method == "sigma":
        n = 3.0 if value is None else float(value)
        return float(np.mean(scores) + n * np.std(scores))
    if method == "fixed":
        if value is None:
            raise ValueError("fixed thresholding requires a value")
        return float(value)
    raise ValueError(f"unknown threshold method {method!r}; choose from {sorted(THRESHOLD_METHODS)}")
