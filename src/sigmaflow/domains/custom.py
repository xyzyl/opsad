"""CustomAdapter: build a domain adapter for any instrument in one call."""

from __future__ import annotations

from .base import DomainAdapter

__all__ = ["CustomAdapter"]


class CustomAdapter(DomainAdapter):
    """A DomainAdapter assembled from plain data — no subclassing needed.

    >>> adapter = CustomAdapter(
    ...     domain="seismology",
    ...     instrument_profiles={
    ...         "broadband_seismometer": {
    ...             "expected_range": {"velocity": (-1e-3, 1e-3)},
    ...             "sample_rate": 100,
    ...         }
    ...     },
    ...     anomaly_categories=["earthquake", "noise_burst", "sensor_glitch"],
    ...     default_features=["spectral_entropy", "kurtosis", "dominant_frequency"],
    ...     category_map={"spike": "sensor_glitch", "noisy": "earthquake",
    ...                   "shift": "earthquake", "flat": "sensor_glitch"},
    ... )
    """

    def __init__(
        self,
        domain: str,
        instrument_profiles: dict | None = None,
        anomaly_categories: list[str] | None = None,
        default_features: list[str] | None = None,
        category_map: dict[str, str] | None = None,
    ):
        self.domain = domain
        self.instrument_profiles = dict(instrument_profiles or {})
        self.anomaly_categories = list(anomaly_categories or [])
        self._default_features = list(default_features) if default_features else None
        self.category_map = dict(category_map or {})

    def feature_defaults(self):
        if self._default_features:
            return list(self._default_features)
        return super().feature_defaults()
