"""DomainAdapter base: inject instrument knowledge into the pipeline."""

from __future__ import annotations

import numpy as np

from ..core.anomaly_result import AnomalyResult, labels_to_intervals
from ..core.base import BaseDetector, BasePreprocessor
from ..core.pipeline import Pipeline
from ..core.signal_frame import SignalFrame, time_to_seconds

__all__ = ["DomainAdapter", "characterize_interval"]


def characterize_interval(values: np.ndarray, i0: int, i1: int) -> str:
    """Classify what the signal did inside an interval:
    'spike', 'flat', 'shift', or 'noisy'."""
    seg = values[i0 : i1 + 1]
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med))) or float(np.std(values)) or 1.0
    scale = 1.4826 * mad
    if len(seg) >= 4 and np.ptp(seg) < 1e-12 * max(1.0, abs(float(seg[0]))):
        return "flat"
    if len(seg) <= 3:
        return "spike"
    if float(np.std(seg)) > 3.0 * scale:
        return "noisy"
    return "shift"


class DomainAdapter:
    """Expert knowledge about one class of instruments.

    Adapters are optional — everything works without them — but they
    reduce false alarms and make detections interpretable: recommended
    preprocessing and detectors, physical-plausibility validation, and
    classification of detected anomalies into named categories.
    """

    domain: str = "generic"
    instrument_profiles: dict = {}
    anomaly_categories: list[str] = []
    #: maps event shape ('spike'/'flat'/'shift'/'noisy') -> category name
    category_map: dict[str, str] = {}

    # ------------------------------------------------------------ validation
    def validate(self, sf: SignalFrame) -> list[str]:
        """Return warnings about physically implausible or suspect data."""
        warnings = []
        profile = self.instrument_profiles.get(sf.instrument or "", {})
        ranges = dict(profile.get("expected_range", {}))
        ranges.update(sf.metadata.get("expected_range", {}))
        for channel in sf.channels:
            x = sf[channel].to_numpy()
            nan_frac = float(np.mean(np.isnan(x)))
            if nan_frac > 0.10:
                warnings.append(
                    f"{channel}: {100 * nan_frac:.0f}% of samples are missing"
                )
            if channel in ranges:
                lo, hi = ranges[channel]
                bad = np.nonzero((x < lo) | (x > hi))[0]
                if len(bad):
                    warnings.append(
                        f"{channel}: {len(bad)} value(s) outside expected range "
                        f"[{lo:g}, {hi:g}] (first at index {int(bad[0])}, "
                        f"value {x[bad[0]]:g})"
                    )
        expected_rate = profile.get("sample_rate")
        if expected_rate and sf.sample_rate:
            ratio = sf.sample_rate / expected_rate
            if not 0.5 <= ratio <= 2.0:
                warnings.append(
                    f"sample rate {sf.sample_rate:g} Hz differs from the "
                    f"{sf.instrument} profile ({expected_rate:g} Hz)"
                )
        return warnings

    # ------------------------------------------------------------ defaults
    def preprocess_defaults(self) -> list[BasePreprocessor]:
        from ..preprocess import GapHandler, Normalizer

        return [GapHandler(), Normalizer(method="robust")]

    def detector_defaults(self) -> BaseDetector:
        from ..detectors import IsolationForestDetector

        return IsolationForestDetector(contamination=0.02,
                                       features=self.feature_defaults())

    def feature_defaults(self) -> list[str]:
        return ["value", "mean", "std", "min", "max", "slope"]

    def default_pipeline(self) -> Pipeline:
        return Pipeline([*self.preprocess_defaults(), self.detector_defaults()])

    # ------------------------------------------------------------ enrichment
    def known_artifact_filter(self, sf: SignalFrame) -> SignalFrame:
        """Remove or flag known instrument artifacts. Default: pass-through."""
        return sf

    def _refine_category(self, category: str, character: str,
                         i0: int, i1: int, sf: SignalFrame) -> str:
        """Hook for adapters that split a shape into finer categories."""
        return category

    def classify_anomaly(self, result: AnomalyResult, sf: SignalFrame,
                         channel: str | None = None) -> AnomalyResult:
        """Attach a domain category to every detected interval.

        Sets ``result.categories`` (one per interval) and
        ``result.classified_intervals`` (dicts with start/end/severity/
        category). The classification is heuristic — shape-based — and
        meant as a first-pass triage, not a verdict.
        """
        channel = channel or sf.channels[0]
        values = sf[channel].to_numpy()
        categories, enriched = [], []
        for start, end, severity in result.intervals:
            i0 = int(sf.time.get_indexer([start])[0])
            i1 = int(sf.time.get_indexer([end])[0])
            character = characterize_interval(values, i0, i1)
            category = self.category_map.get(character, "unclassified")
            category = self._refine_category(category, character, i0, i1, sf)
            categories.append(category)
            enriched.append({"start": start, "end": end,
                             "severity": severity, "category": category})
        result.categories = categories
        result.classified_intervals = enriched
        return result

    def __repr__(self) -> str:
        return f"{type(self).__name__}(domain={self.domain!r})"


def _duration_fraction(i0: int, i1: int, sf: SignalFrame) -> float:
    secs = time_to_seconds(sf.time)
    total = secs[-1] - secs[0]
    return (secs[i1] - secs[i0]) / total if total > 0 else 0.0
