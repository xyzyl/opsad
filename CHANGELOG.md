# Changelog

## [0.1.0] - Unreleased

Initial MVP release.

- `SignalFrame` time-series data model with units, sample rate, and metadata.
- `AnomalyResult` detection output with scores, labels, intervals, and plotting.
- Statistical detectors: `ZScoreDetector`, `ModifiedZScoreDetector`, `CUSUMDetector`, `STLResidualDetector`.
- ML detectors: `IsolationForestDetector`, `LOFDetector` (windowed feature extraction).
- Preprocessors: `Resampler`, `GapHandler`, `Detrend`, `Normalizer`.
- `Pipeline` composition with YAML save/load.
- Evaluation metrics: point-based, event-based, and threshold-independent.
- Synthetic signal generators for plasma and ocean domains.
- CLI: `sigmaflow detect`, `sigmaflow evaluate`, `sigmaflow dashboard`.
- Interactive Dash dashboard (`sigmaflow[dashboard]` extra): signal browser,
  live detector tuning, threshold control without re-scoring, ground-truth
  overlay, click-to-zoom anomaly table.
- Plain-language interpretation panel in the dashboard: explains the current
  signal, detector, and threshold in lay terms, characterizes the strongest
  detected event (spike / flat-line / sustained shift / noise burst), and maps
  it to its domain-specific and cross-domain significance.
