# Ocean tutorial: quality-controlling a live NOAA buoy

NOAA's National Data Buoy Center publishes moored-buoy measurements in near
real time. This tutorial pulls the last ~45 days from a buoy and separates
real ocean events from sensor artifacts.

## 1. Fetch live data

```python
from sigmaflow.data import fetch_ndbc_buoy

signal = fetch_ndbc_buoy("46042")    # Monterey Bay; any NDBC station id works
signal.summary()
# channels: water_temperature [°C], air_temperature [°C], pressure [hPa]
```

## 2. Validate against ocean physics

```python
from sigmaflow.domains import OceanAdapter

adapter = OceanAdapter()
print(adapter.validate(signal))
# flags temperatures below -2.5 °C (colder than seawater can be), heavy gaps, ...
```

## 3. Remove instrument artifacts, then detect

```python
clean = adapter.known_artifact_filter(signal)   # Hampel despike (wave slap, debris)
pipeline = adapter.default_pipeline()            # gap fill -> robust norm -> STL residual
result = pipeline.fit_detect(clean)
result.summary()
```

The ocean default detector is `STLResidualDetector`: buoy records carry
strong daily cycles, and removing them first stops every warm afternoon from
alarming.

## 4. Classify: event or sensor fault?

```python
result = adapter.classify_anomaly(result, clean)
print(result.categories)
# physical_event  -> short excursion (heatwave-like)
# sensor_drift    -> a shift persisting >25% of the record (biofouling-like)
# telemetry_dropout -> flat-lined stretches
```

## 5. Watch for drift as data arrives

```python
from sigmaflow import StreamingDetector
from sigmaflow.detectors import CUSUMDetector

detector = CUSUMDetector(threshold=15.0, drift=1.0).fit(clean)
stream = StreamingDetector(detector, context_size=2000)
# each new telemetry batch:
result = stream.update(new_times, new_values)
if result.n_anomalies:
    print("drift alarm:", result.intervals)
```
