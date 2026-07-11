"""Ocean sensor adapter (CTD, ARGO floats, moored buoys, tide gauges)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.signal_frame import SignalFrame
from .base import DomainAdapter, _duration_fraction

__all__ = ["OceanAdapter"]


class OceanAdapter(DomainAdapter):
    domain = "ocean"

    instrument_profiles = {
        "ctd": {
            "expected_range": {"temperature": (-2.5, 40.0), "salinity": (0.0, 42.0)},
            "known_artifacts": ["biofouling_drift", "pressure_drift", "salinity_offset"],
        },
        "argo_float": {
            "expected_range": {"temperature": (-2.5, 40.0), "salinity": (0.0, 42.0)},
            "known_artifacts": ["biofouling_drift", "telemetry_dropout"],
        },
        "moored_buoy": {
            "expected_range": {"water_temperature": (-2.5, 40.0),
                               "air_temperature": (-60.0, 60.0),
                               "pressure": (870.0, 1085.0)},
            "known_artifacts": ["wave_noise", "debris_interference", "solar_heating"],
        },
        "tide_gauge": {
            "known_artifacts": ["datum_shift", "harbor_seiching"],
        },
    }

    anomaly_categories = [
        "sensor_drift", "biofouling", "telemetry_dropout",
        "physical_event", "density_inversion",
    ]

    category_map = {
        "spike": "physical_event",
        "flat": "telemetry_dropout",
        "shift": "physical_event",
        "noisy": "physical_event",
    }

    def _refine_category(self, category, character, i0, i1, sf):
        # a shift persisting for a large fraction of the record reads as
        # drift (biofouling-like), not a discrete ocean event
        if character == "shift" and _duration_fraction(i0, i1, sf) > 0.25:
            return "sensor_drift"
        return category

    def feature_defaults(self):
        return ["mean", "std", "slope", "range", "iqr"]

    def preprocess_defaults(self):
        from ..preprocess import GapHandler, Normalizer

        return [
            GapHandler(max_gap="3h", fill_method="interpolate"),
            Normalizer(method="robust"),
        ]

    def detector_defaults(self):
        from ..detectors import STLResidualDetector

        # ocean records carry strong diurnal/seasonal rhythm
        return STLResidualDetector(residual_threshold=4.0)

    def known_artifact_filter(self, sf: SignalFrame) -> SignalFrame:
        """Despike with a Hampel filter: samples more than 5 robust sigmas
        from their rolling median are replaced by that median (wave slap,
        debris hits, seabirds on the sensor)."""
        df = sf.values.copy()
        window = 11
        replaced = {}
        for c in sf.channels:
            s = df[c]
            med = s.rolling(window, center=True, min_periods=3).median()
            mad = (s - med).abs().rolling(window, center=True, min_periods=3).median()
            scale = 1.4826 * mad.replace(0.0, np.nan)
            outliers = ((s - med).abs() / scale) > 5.0
            outliers = outliers.fillna(False)
            if outliers.any():
                replaced[c] = int(outliers.sum())
                df[c] = s.where(~outliers, med)
        if not replaced:
            return sf
        metadata = dict(sf.metadata)
        metadata["despiked"] = replaced
        return sf._with(df, metadata=metadata)

    def validate(self, sf: SignalFrame) -> list[str]:
        warnings = super().validate(sf)
        # ARGO-style density-inversion proxy: temperature and salinity
        # jointly implausible is out of scope without pressure, but flag
        # sub-freezing seawater explicitly
        for channel in sf.channels:
            if "temp" in channel.lower():
                x = sf[channel].to_numpy()
                frozen = np.nonzero(x < -2.5)[0]
                if len(frozen):
                    warnings.append(
                        f"{channel}: {len(frozen)} reading(s) below -2.5 °C — "
                        "colder than seawater can be; likely sensor error"
                    )
        return warnings
