"""Abstract base classes for detectors and preprocessors."""

from __future__ import annotations

import inspect
import time as _time
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from .anomaly_result import AnomalyResult
from .signal_frame import SignalFrame

__all__ = ["ParamsMixin", "BaseDetector", "BasePreprocessor"]


class ParamsMixin:
    """sklearn-style parameter introspection.

    Subclasses must store each ``__init__`` argument on an instance
    attribute of the same name; ``get_params`` reads them back. This is
    what makes pipelines serializable to YAML.
    """

    def get_params(self) -> dict[str, Any]:
        sig = inspect.signature(type(self).__init__)
        params = {}
        for name in sig.parameters:
            if name in ("self", "args", "kwargs"):
                continue
            if hasattr(self, name):
                params[name] = getattr(self, name)
        return params

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.get_params().items())
        return f"{type(self).__name__}({args})"


class BaseDetector(ParamsMixin, ABC):
    """Common interface for all detectors: fit / score / detect.

    Scores are per-timestep floats where higher means more anomalous.
    Multichannel signals are scored per channel and aggregated with the
    pointwise maximum: a timestep is as anomalous as its worst channel.
    """

    name: str = "base"

    def __init__(self):
        self.threshold_method: str = "auto"
        self.threshold_value: float | None = None
        self.threshold_: float | None = None  # resolved at detect() time
        self._fitted = False

    # ---------------------------------------------------------------- #
    # Channel-level hooks for subclasses
    # ---------------------------------------------------------------- #

    def _fit_channel(self, x: np.ndarray, channel: str, sf: SignalFrame) -> None:
        """Learn per-channel state from training data. Default: nothing."""

    @abstractmethod
    def _score_channel(self, x: np.ndarray, channel: str, sf: SignalFrame) -> np.ndarray:
        """Return per-timestep anomaly scores for one channel."""

    # ---------------------------------------------------------------- #
    # Public API
    # ---------------------------------------------------------------- #

    def fit(self, sf: SignalFrame) -> "BaseDetector":
        """Learn normal behavior from a (presumed normal) training signal."""
        for c in sf.channels:
            self._fit_channel(sf[c].to_numpy(), c, sf)
        self._fitted = True
        return self

    def score(self, sf: SignalFrame) -> np.ndarray:
        """Per-timestep anomaly scores, aggregated across channels (max)."""
        per_channel = [
            np.nan_to_num(
                np.asarray(self._score_channel(sf[c].to_numpy(), c, sf), dtype=np.float64),
                nan=0.0, posinf=0.0, neginf=0.0,
            )
            for c in sf.channels
        ]
        return np.max(np.vstack(per_channel), axis=0)

    def detect(self, sf: SignalFrame) -> AnomalyResult:
        """Score the signal and threshold into binary anomaly labels."""
        t0 = _time.perf_counter()
        scores = self.score(sf)
        threshold = self._resolve_threshold(scores)
        self.threshold_ = threshold
        labels = (scores > threshold).astype(int)
        elapsed = _time.perf_counter() - t0
        return AnomalyResult(
            labels=labels,
            scores=scores,
            threshold=threshold,
            detector_name=self.name,
            parameters=self.get_params(),
            signal=sf,
            computation_time=elapsed,
        )

    def fit_detect(self, sf: SignalFrame) -> AnomalyResult:
        return self.fit(sf).detect(sf)

    def set_threshold(self, method: str = "auto", value: float | None = None) -> "BaseDetector":
        """Choose the thresholding strategy: auto, percentile, sigma, or fixed."""
        from ..detectors.threshold import THRESHOLD_METHODS

        if method not in THRESHOLD_METHODS:
            raise ValueError(
                f"unknown threshold method {method!r}; choose from {sorted(THRESHOLD_METHODS)}"
            )
        self.threshold_method = method
        self.threshold_value = value
        return self

    # ---------------------------------------------------------------- #
    # Thresholding
    # ---------------------------------------------------------------- #

    def _auto_threshold(self, scores: np.ndarray) -> float:
        """Detector-specific default threshold. Override in subclasses."""
        return float(np.mean(scores) + 3.0 * np.std(scores))

    def _resolve_threshold(self, scores: np.ndarray) -> float:
        from ..detectors.threshold import compute_threshold

        if self.threshold_method == "auto":
            return self._auto_threshold(scores)
        return compute_threshold(scores, self.threshold_method, self.threshold_value)


class BasePreprocessor(ParamsMixin, ABC):
    """Common interface for all preprocessing steps."""

    name: str = "base"

    def fit(self, sf: SignalFrame) -> "BasePreprocessor":
        """Learn transform state from data. Default: stateless, nothing to fit."""
        return self

    @abstractmethod
    def transform(self, sf: SignalFrame) -> SignalFrame:
        """Apply the transform, returning a new SignalFrame."""

    def fit_transform(self, sf: SignalFrame) -> SignalFrame:
        return self.fit(sf).transform(sf)

    def inverse_transform(self, sf: SignalFrame) -> SignalFrame:
        raise NotImplementedError(f"{type(self).__name__} is not invertible")
