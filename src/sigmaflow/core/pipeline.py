"""Pipeline: composable preprocessing + detection with YAML serialization."""

from __future__ import annotations

from typing import Any, Sequence

import yaml

from .anomaly_result import AnomalyResult
from .base import BaseDetector, BasePreprocessor
from .signal_frame import SignalFrame

__all__ = ["Pipeline", "component_registry"]


def component_registry() -> dict[str, type]:
    """Map component names (class ``name`` attribute) to classes.

    Imported lazily so ``core`` never depends on the concrete modules at
    import time.
    """
    from ..detectors import DETECTOR_REGISTRY
    from ..preprocess import Detrend, GapHandler, Normalizer, Resampler

    registry: dict[str, type] = dict(DETECTOR_REGISTRY)
    for cls in (Resampler, GapHandler, Detrend, Normalizer):
        registry[cls.name] = cls
    return registry


def _yaml_safe(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_yaml_safe(v) for v in value]
    if isinstance(value, list):
        return [_yaml_safe(v) for v in value]
    if isinstance(value, dict):
        return {k: _yaml_safe(v) for k, v in value.items()}
    return value


class Pipeline:
    """An ordered chain of preprocessors, optionally ending in a detector.

    ``fit_detect(sf)`` runs the whole chain; ``save``/``load`` serialize
    the configuration (not fitted state) to YAML for sharing and
    reproducibility.
    """

    def __init__(self, steps: Sequence[BasePreprocessor | BaseDetector]):
        steps = list(steps)
        if not steps:
            raise ValueError("Pipeline needs at least one step")
        for step in steps[:-1]:
            if isinstance(step, BaseDetector):
                raise ValueError(
                    "a detector may only appear as the final pipeline step "
                    f"(found {type(step).__name__} earlier)"
                )
            if not isinstance(step, BasePreprocessor):
                raise TypeError(f"{type(step).__name__} is not a preprocessor")
        last = steps[-1]
        if not isinstance(last, (BasePreprocessor, BaseDetector)):
            raise TypeError(f"{type(last).__name__} is not a preprocessor or detector")
        self.steps = steps

    @property
    def detector(self) -> BaseDetector | None:
        return self.steps[-1] if isinstance(self.steps[-1], BaseDetector) else None

    @property
    def preprocessors(self) -> list[BasePreprocessor]:
        return [s for s in self.steps if isinstance(s, BasePreprocessor)]

    # ---------------------------------------------------------------- #
    # Execution
    # ---------------------------------------------------------------- #

    def fit(self, sf: SignalFrame) -> "Pipeline":
        current = sf
        for step in self.preprocessors:
            current = step.fit_transform(current)
        if self.detector is not None:
            self.detector.fit(current)
        return self

    def transform(self, sf: SignalFrame) -> SignalFrame:
        current = sf
        for step in self.preprocessors:
            current = step.transform(current)
        return current

    def detect(self, sf: SignalFrame) -> AnomalyResult:
        if self.detector is None:
            raise ValueError("this pipeline has no detector as its final step")
        return self.detector.detect(self.transform(sf))

    def fit_detect(self, sf: SignalFrame) -> AnomalyResult:
        return self.fit(sf).detect(sf)

    # ---------------------------------------------------------------- #
    # Serialization
    # ---------------------------------------------------------------- #

    def to_config(self) -> dict:
        return {
            "sigmaflow_pipeline": 1,
            "steps": [
                {"class": step.name, "params": _yaml_safe(step.get_params())}
                for step in self.steps
            ],
        }

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_config(), f, sort_keys=False)

    @classmethod
    def from_config(cls, config: dict) -> "Pipeline":
        registry = component_registry()
        steps = []
        for entry in config["steps"]:
            name = entry["class"]
            if name not in registry:
                raise ValueError(f"unknown pipeline component {name!r}")
            steps.append(registry[name](**(entry.get("params") or {})))
        return cls(steps)

    @classmethod
    def load(cls, path: str) -> "Pipeline":
        with open(path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        if not isinstance(config, dict) or "steps" not in config:
            raise ValueError(f"{path} is not a sigmaflow pipeline file")
        return cls.from_config(config)

    def __repr__(self) -> str:
        inner = ",\n  ".join(repr(s) for s in self.steps)
        return f"Pipeline([\n  {inner}\n])"
