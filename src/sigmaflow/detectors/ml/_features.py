"""Backwards-compatible shim: the minimal MVP extractor now lives in
:mod:`sigmaflow.features.extractors` as the configurable FeatureExtractor."""

from __future__ import annotations

import numpy as np

from ...features.extractors import DEFAULT_FEATURES, FeatureExtractor

__all__ = ["window_features", "FEATURE_NAMES"]

FEATURE_NAMES = list(DEFAULT_FEATURES)


def window_features(x: np.ndarray, window_size: int) -> np.ndarray:
    """Default six-feature window matrix (value/mean/std/min/max/slope)."""
    return FeatureExtractor(window_size=window_size).transform(x)
