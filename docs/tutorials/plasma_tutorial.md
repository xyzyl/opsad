# Plasma tutorial: anomalies in live solar wind data

The solar wind is a plasma — protons streaming from the Sun past Earth at
hundreds of km/s. NOAA measures it in real time ~1.5 million km upstream.
This tutorial finds regime changes (shocks, stream interfaces) in today's data.

## 1. Fetch live data

```python
from sigmaflow.data import fetch_solar_wind

signal = fetch_solar_wind()          # no API key needed
signal.summary()
# channels: proton_density [cm^-3], proton_speed [km/s], proton_temperature [K]
```

## 2. Let the adapter validate it

```python
from sigmaflow.domains import PlasmaAdapter

adapter = PlasmaAdapter()
for warning in adapter.validate(signal):
    print("!", warning)
```

The adapter knows plausible ranges for solar-wind plasma (density 0.1–150
cm⁻³, speed 200–1200 km/s) and flags telemetry glitches before they pollute
detection.

## 3. Detect with the adapter's recommended pipeline

```python
pipeline = adapter.default_pipeline()   # gap fill -> detrend -> robust norm -> iforest
result = pipeline.fit_detect(signal)
result.summary()
```

The plasma defaults lean on spectral features (`spectral_entropy`,
`dominant_frequency`) because plasma instabilities have characteristic
frequencies.

## 4. Classify what was found

```python
result = adapter.classify_anomaly(result, signal)
for interval in result.classified_intervals:
    print(interval["start"], "->", interval["category"])
# e.g. disruption_precursor (sustained regime change), rf_interference (spikes)
```

A sustained shift in solar-wind data usually means a stream interface or CME
front arriving — the same mathematical signature a tokamak team watches for
as a disruption precursor.

## 5. Go deeper: sequence-aware detection

```python
from sigmaflow.detectors import LSTMAutoencoderDetector   # pip install sigmaflow[deep]

deep = LSTMAutoencoderDetector(window_size=64, epochs=10)
deep_result = deep.fit(signal.slice(signal.time[0], signal.time[len(signal)//2])) \
                  .detect(signal)
```

The LSTM autoencoder learns the normal *sequence* of fluctuations; fronts and
shocks reconstruct poorly because their ordering is unfamiliar.
