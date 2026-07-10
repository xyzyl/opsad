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

__all__ = [
    "ZScoreDetector",
    "ModifiedZScoreDetector",
    "CUSUMDetector",
    "STLResidualDetector",
    "IsolationForestDetector",
    "LOFDetector",
    "DETECTOR_REGISTRY",
    "THRESHOLD_METHODS",
    "compute_threshold",
]
