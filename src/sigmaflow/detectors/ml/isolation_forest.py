"""Isolation Forest detector on windowed features."""

from __future__ import annotations

from sklearn.ensemble import IsolationForest

from .base import WindowedMLDetector

__all__ = ["IsolationForestDetector"]


class IsolationForestDetector(WindowedMLDetector):
    """Isolation Forest over per-timestep window features.

    Each timestep becomes a feature vector (configurable via
    ``features``; defaults to value/mean/std/min/max/slope); anomalies
    are isolated in fewer random splits. Scores are the negated sklearn
    decision function, so scores above 0 are what sklearn itself would
    label anomalous at the configured ``contamination`` — which is why
    the auto threshold is 0.
    """

    name = "isolation_forest"

    def __init__(
        self,
        n_estimators: int = 200,
        contamination="auto",
        window_size: int = 25,
        features: list[str] | None = None,
        multivariate: bool = False,
        random_state: int | None = 0,
    ):
        super().__init__(window_size=window_size, features=features,
                         multivariate=multivariate)
        self.n_estimators = int(n_estimators)
        self.contamination = contamination
        self.random_state = random_state

    def _make_model(self):
        return IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
        )
