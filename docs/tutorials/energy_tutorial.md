# Energy tutorial: Great Britain grid frequency

Grid frequency is the heartbeat of a power system: nominally 50 Hz in GB,
dropping when demand outruns supply and rising when supply outruns demand.
Elexon publishes GB frequency at 15-second resolution, no key required.

## 1. Fetch the last 24 hours

```python
from sigmaflow.data import fetch_gb_grid_frequency

signal = fetch_gb_grid_frequency(hours=24)
signal.summary()
```

## 2. Validate against the operating band

```python
from sigmaflow.domains import EnergyGridAdapter

adapter = EnergyGridAdapter()
print(adapter.validate(signal))
# reports any samples beyond ±0.5 Hz of nominal — genuine operating-band excursions
```

## 3. Detect excursions

```python
result = adapter.default_pipeline().fit_detect(signal)
result = adapter.classify_anomaly(result, signal)
for iv in result.classified_intervals:
    print(iv["start"], iv["category"])
```

The energy adapter's classification uses physics: a *sustained* deviation
below nominal is labeled `generation_dropout` (the grid lost supply), above
nominal `load_spike` (it lost demand); sharp transients are
`frequency_excursion`.

## 4. Ensemble: catch spikes and drifts at once

```python
from sigmaflow.detectors import (
    CUSUMDetector, EnsembleDetector, ModifiedZScoreDetector,
)

ensemble = EnsembleDetector(
    detectors=[ModifiedZScoreDetector(window_size=240),  # sharp excursions
               CUSUMDetector(threshold=10.0)],           # slow imbalance
    aggregation="mean",
).set_threshold("fixed", 3.0)   # 3 robust sigmas of the combined score
result = ensemble.fit_detect(signal)
```

## 5. Benchmark your setup

```python
from sigmaflow.evaluation import load_nab, compare_detectors

nab = load_nab("realKnownCause/nyc_taxi.csv")   # labeled real-world series
comparison = compare_detectors(nab, [ensemble, adapter.detector_defaults()])
print(comparison.to_markdown())
```
