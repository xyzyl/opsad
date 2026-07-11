"""Satellite telemetry adapter (attitude, power, thermal, comms, fields)."""

from __future__ import annotations

from .base import DomainAdapter

__all__ = ["SatelliteAdapter"]


class SatelliteAdapter(DomainAdapter):
    domain = "satellite"

    instrument_profiles = {
        "goes_magnetometer": {
            "expected_range": {"Hp": (-500.0, 500.0), "He": (-500.0, 500.0),
                               "Hn": (-500.0, 500.0), "total": (0.0, 600.0)},
            "known_artifacts": ["arcjet_firings", "eclipse_transients"],
        },
        "power_subsystem": {
            "known_artifacts": ["eclipse_artifact"],
        },
        "star_tracker": {
            "known_artifacts": ["sun_blinding", "moon_in_fov"],
        },
    }

    anomaly_categories = [
        "safe_mode_trigger", "power_anomaly", "thermal_excursion",
        "attitude_disturbance", "eclipse_artifact",
    ]

    # heuristic first-pass triage by event shape; telemetry with orbital
    # context can refine this far better
    category_map = {
        "spike": "attitude_disturbance",
        "flat": "safe_mode_trigger",
        "shift": "thermal_excursion",
        "noisy": "attitude_disturbance",
    }

    def feature_defaults(self):
        return ["mean", "std", "slope", "autocorrelation_lag1", "range"]

    def preprocess_defaults(self):
        from ..preprocess import Detrend, GapHandler, Normalizer

        return [
            GapHandler(max_gap="10min", fill_method="interpolate"),
            # remove slow orbital variation; keep sub-orbital structure
            Detrend(method="moving_average", window=181),
            Normalizer(method="robust"),
        ]

    def detector_defaults(self):
        from ..detectors import IsolationForestDetector

        return IsolationForestDetector(contamination=0.02, window_size=31,
                                       features=self.feature_defaults())
