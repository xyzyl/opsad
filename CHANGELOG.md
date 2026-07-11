# Changelog

## [0.2.0] - Unreleased

Live data, the full detector suite, and domain intelligence.

- **Live open-data sources** (`sigmaflow.data`): NOAA real-time solar wind
  (plasma), NOAA NDBC buoys (ocean), GOES magnetometer (satellite), Elexon GB
  grid frequency (energy) — public feeds, no API keys, attribution and fetch
  timestamps in metadata. Dashboard and static-site export now run on real
  data; synthetic generators remain for testing only.
- **Deep learning detectors** (`sigmaflow[deep]`, PyTorch):
  `AutoencoderDetector` (fc / conv1d), `LSTMAutoencoderDetector`,
  `TransformerDetector` — denoising reconstruction with percentile
  thresholds.
- **EnsembleDetector**: mean / max / voting / weighted aggregation with
  robust per-member score standardization.
- **Domain adapters** (`sigmaflow.domains`): plasma, ocean, satellite,
  energy grid, plus `CustomAdapter`. Physical-plausibility validation,
  recommended pipelines, artifact filters (saturation flagging, Hampel
  despike), and shape-based anomaly classification
  (`disruption_precursor`, `sensor_drift`, `generation_dropout`, ...).
- **FeatureExtractor** (`sigmaflow.features`): 17 vectorized windowed
  features; ML detectors take a `features=[...]` list and a
  `multivariate=True` joint-channel mode.
- **Evaluation**: range-based precision/recall/F1 (after Tatbul et al.),
  `compare_detectors` with summary tables (markdown/LaTeX) and ROC/PR/
  detection plots, NAB and SMD benchmark loaders (auto-download + cache).
- **StreamingDetector**: online scoring of arriving chunks with rolling
  context.
- Dashboard: deep detectors in the UI, live signals in the browser
  (synthetic only as offline fallback), data-provenance notes in the
  narrative and static site.

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
- Static site export (`sigmaflow dashboard --export DIR`): a deployable ~1 MB
  static dashboard (Cloudflare Pages, GitHub Pages, any static host) with
  precomputed scores; thresholding, metrics, table, and narrative run
  client-side. See docs/deploying.md.
