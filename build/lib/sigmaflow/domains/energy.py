"""Energy grid adapter (frequency, load, generation, voltage)."""

from __future__ import annotations

import numpy as np

from ..core.signal_frame import SignalFrame
from .base import DomainAdapter

__all__ = ["EnergyGridAdapter"]


class EnergyGridAdapter(DomainAdapter):
    domain = "energy"

    instrument_profiles = {
        "grid_frequency_sensor": {
            "expected_range": {"frequency": (49.0, 51.0)},  # 50 Hz systems
            "known_artifacts": ["measurement_gaps"],
        },
        "load_sensor": {
            "known_artifacts": ["diurnal_pattern", "weekend_effect"],
        },
    }

    anomaly_categories = [
        "frequency_excursion", "load_spike", "generation_dropout",
        "voltage_sag", "line_fault",
    ]

    category_map = {
        "spike": "frequency_excursion",
        "flat": "line_fault",
        "shift": "frequency_excursion",
        "noisy": "line_fault",
    }

    def _refine_category(self, category, character, i0, i1, sf):
        # for frequency channels the *sign* of a sustained deviation says
        # which side of the supply/demand balance moved
        channel = sf.channels[0]
        if character == "shift" and "freq" in channel.lower():
            nominal = sf.metadata.get("nominal_frequency", 50.0)
            seg = sf[channel].to_numpy()[i0 : i1 + 1]
            if np.nanmean(seg) < nominal:
                return "generation_dropout"   # under-frequency: lost supply
            return "load_spike"               # over-frequency: lost demand
        return category

    def feature_defaults(self):
        return ["mean", "std", "zero_crossing_rate", "range", "slope"]

    def preprocess_defaults(self):
        from ..preprocess import Detrend, GapHandler, Normalizer

        return [
            GapHandler(max_gap="5min", fill_method="interpolate"),
            Detrend(method="moving_average", window=241),  # remove slow ramps
            Normalizer(method="robust"),
        ]

    def detector_defaults(self):
        from ..detectors import ModifiedZScoreDetector

        # frequency excursions are sharp; a rolling robust z-score reacts
        # fast without alarming on slow regulation wander
        return ModifiedZScoreDetector(window_size=240, threshold=4.0)

    def validate(self, sf: SignalFrame) -> list[str]:
        warnings = super().validate(sf)
        nominal = sf.metadata.get("nominal_frequency")
        for channel in sf.channels:
            if nominal and "freq" in channel.lower():
                x = sf[channel].to_numpy()
                out = np.nonzero(np.abs(x - nominal) > 0.5)[0]
                if len(out):
                    warnings.append(
                        f"{channel}: {len(out)} sample(s) beyond ±0.5 Hz of "
                        f"nominal {nominal:g} Hz — genuine operating-band excursions"
                    )
        return warnings
