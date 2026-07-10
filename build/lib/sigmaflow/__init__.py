"""sigmaflow: cross-domain anomaly detection for scientific time-series."""

from .core.anomaly_result import AnomalyResult
from .core.pipeline import Pipeline
from .core.signal_frame import SignalFrame

__version__ = "0.1.0"

__all__ = ["SignalFrame", "AnomalyResult", "Pipeline", "detect", "__version__"]


def detect(signal: SignalFrame, method: str = "isolation_forest", **params) -> AnomalyResult:
    """One-call detection with sensible defaults.

    >>> import sigmaflow as sf
    >>> signal = sf.SignalFrame(time=t, values=y)
    >>> result = sf.detect(signal, method="isolation_forest")
    """
    from .detectors import DETECTOR_REGISTRY

    if method not in DETECTOR_REGISTRY:
        raise ValueError(
            f"unknown detection method {method!r}; choose from {sorted(DETECTOR_REGISTRY)}"
        )
    detector = DETECTOR_REGISTRY[method](**params)
    return detector.fit_detect(signal)
