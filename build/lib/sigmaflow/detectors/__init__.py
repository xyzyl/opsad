from .ensemble import EnsembleDetector
from .ml import IsolationForestDetector, LOFDetector
from .statistical import (
    CUSUMDetector,
    ModifiedZScoreDetector,
    STLResidualDetector,
    ZScoreDetector,
)
from .threshold import THRESHOLD_METHODS, compute_threshold

DETECTOR_REGISTRY = {
    "zscore": ZScoreDetector,
    "modified_zscore": ModifiedZScoreDetector,
    "cusum": CUSUMDetector,
    "stl_residual": STLResidualDetector,
    "isolation_forest": IsolationForestDetector,
    "lof": LOFDetector,
}

# deep detectors are optional: pip install sigmaflow[deep]
try:
    from .deep import (  # noqa: F401
        AutoencoderDetector,
        LSTMAutoencoderDetector,
        TransformerDetector,
    )

    DETECTOR_REGISTRY.update({
        "autoencoder": AutoencoderDetector,
        "lstm_autoencoder": LSTMAutoencoderDetector,
        "transformer": TransformerDetector,
    })
    _HAS_DEEP = True
except ImportError:  # torch not installed
    _HAS_DEEP = False

__all__ = [
    "ZScoreDetector",
    "ModifiedZScoreDetector",
    "CUSUMDetector",
    "STLResidualDetector",
    "IsolationForestDetector",
    "LOFDetector",
    "EnsembleDetector",
    "DETECTOR_REGISTRY",
    "THRESHOLD_METHODS",
    "compute_threshold",
]
if _HAS_DEEP:
    __all__ += ["AutoencoderDetector", "LSTMAutoencoderDetector", "TransformerDetector"]
