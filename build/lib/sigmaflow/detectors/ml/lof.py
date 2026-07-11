"""Local Outlier Factor detector on windowed features."""

from __future__ import annotations

from sklearn.neighbors import LocalOutlierFactor

from .base import WindowedMLDetector

__all__ = ["LOFDetector"]


class LOFDetector(WindowedMLDetector):
    """Local Outlier Factor over per-timestep window features.

    Density-based: a timestep is anomalous when its feature-space
    neighborhood is sparser than its neighbors' — catching values that
    are locally unusual even when globally in range. Scores are the
    negated sklearn decision function (auto threshold 0, matching
    sklearn's own contamination-based labeling).
    """

    name = "lof"

    def __init__(
        self,
        n_neighbors: int = 20,
        contamination="auto",
        window_size: int = 25,
        features: list[str] | None = None,
        multivariate: bool = False,
    ):
        super().__init__(window_size=window_size, features=features,
                         multivariate=multivariate)
        self.n_neighbors = int(n_neighbors)
        self.contamination = contamination
        self._n_samples_hint: int | None = None

    def _make_model(self):
        n_neighbors = self.n_neighbors
        if self._n_samples_hint is not None:
            n_neighbors = max(1, min(n_neighbors, self._n_samples_hint - 1))
        return LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=self.contamination,
            novelty=True,
        )

    def _fit_channel(self, x, channel, sf):
        self._n_samples_hint = len(x)
        super()._fit_channel(x, channel, sf)

    def fit(self, sf):
        self._n_samples_hint = len(sf)
        return super().fit(sf)
