"""Plasma diagnostics adapter (Langmuir probes, magnetics, interferometry)."""

from __future__ import annotations

import numpy as np

from ..core.signal_frame import SignalFrame
from .base import DomainAdapter, _duration_fraction

__all__ = ["PlasmaAdapter"]


class PlasmaAdapter(DomainAdapter):
    domain = "plasma"

    instrument_profiles = {
        "langmuir_probe": {
            "expected_range": {"n_e": (1e16, 1e21), "T_e": (0.1, 10000.0)},
            "known_artifacts": ["probe_saturation", "sheath_expansion", "rf_pickup"],
        },
        "mirnov_coil": {
            "known_artifacts": ["integrator_drift", "resonance_saturation"],
        },
        "interferometer": {
            "known_artifacts": ["fringe_jumps", "vibration_noise"],
        },
        "solar_wind_plasma_analyzer": {
            "expected_range": {"proton_density": (0.1, 150.0),
                               "proton_speed": (200.0, 1200.0),
                               "proton_temperature": (1e3, 2e6)},
            "known_artifacts": ["telemetry_gaps", "instrument_mode_changes"],
        },
    }

    anomaly_categories = [
        "disruption_precursor", "elm_event", "sensor_saturation",
        "calibration_drift", "rf_interference",
    ]

    category_map = {
        "spike": "rf_interference",
        "flat": "sensor_saturation",
        "shift": "disruption_precursor",
        "noisy": "rf_interference",
    }

    def _refine_category(self, category, character, i0, i1, sf):
        # a slow shift over a large fraction of the record is drift,
        # not a fast-growing instability
        if character == "shift" and _duration_fraction(i0, i1, sf) > 0.25:
            return "calibration_drift"
        return category

    def feature_defaults(self):
        # MHD modes have characteristic frequencies: lean on spectral features
        return ["std", "range", "spectral_entropy", "dominant_frequency",
                "zero_crossing_rate"]

    def preprocess_defaults(self):
        from ..preprocess import Detrend, GapHandler, Normalizer

        return [
            GapHandler(max_gap="1s", fill_method="interpolate"),
            Detrend(method="moving_average", window=501),
            Normalizer(method="robust"),
        ]

    def detector_defaults(self):
        from ..detectors import IsolationForestDetector

        return IsolationForestDetector(contamination=0.02, window_size=51,
                                       features=self.feature_defaults())

    def known_artifact_filter(self, sf: SignalFrame) -> SignalFrame:
        """Flag saturation plateaus (flat-topped stretches at the channel
        maximum) as NaN so detectors don't chase instrument ceilings."""
        df = sf.values.copy()
        flagged = {}
        for c in sf.channels:
            x = df[c].to_numpy().copy()
            top = np.nanmax(x)
            at_top = x >= top * (1 - 1e-9)
            # runs of >= 5 samples pinned at the maximum are saturation
            run = 0
            n_flagged = 0
            for i in range(len(x) + 1):
                if i < len(x) and at_top[i]:
                    run += 1
                    continue
                if run >= 5:
                    x[i - run : i] = np.nan
                    n_flagged += run
                run = 0
            if n_flagged:
                flagged[c] = n_flagged
                df[c] = x
        if not flagged:
            return sf
        metadata = dict(sf.metadata)
        metadata["saturation_flagged"] = flagged
        return sf._with(df, metadata=metadata)
