"""Synthetic but physically flavored benchmark signals with ground-truth labels."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.signal_frame import SignalFrame

__all__ = [
    "generate_plasma_signal",
    "generate_ocean_temperature",
    "generate_generic_signal",
]


def generate_plasma_signal(
    duration: float = 5.0,
    sample_rate: float = 10_000.0,
    anomalies: list[dict] | None = None,
    seed: int | None = 0,
) -> SignalFrame:
    """Langmuir-probe-like electron density signal.

    Baseline: slowly modulated density around 1e19 m^-3 with an MHD-like
    kHz oscillation and measurement noise. Injectable anomalies:

    - ``{"type": "disruption_precursor", "onset": t, "growth_rate": g}`` —
      exponentially growing oscillation from ``onset`` to the end.
    - ``{"type": "sensor_saturation", "start": t0, "end": t1}`` — signal
      clipped flat at the instrument maximum.
    - ``{"type": "spike", "time": t, "magnitude": m}`` — single-point spike
      of ``m`` times the baseline density.

    If ``anomalies`` is None, one precursor and one saturation are injected.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, duration, 1.0 / sample_rate)
    n = len(t)

    n0 = 1e19
    x = n0 * (1.0 + 0.05 * np.sin(2 * np.pi * 0.5 * t))          # slow modulation
    x += n0 * 0.01 * np.sin(2 * np.pi * 2000.0 * t)              # MHD-like mode
    x += rng.normal(0.0, 0.01 * n0, n)                           # noise

    labels = np.zeros(n, dtype=int)
    if anomalies is None:
        anomalies = [
            {"type": "disruption_precursor", "onset": 0.64 * duration, "growth_rate": 8.0 / duration},
            {"type": "sensor_saturation", "start": 0.90 * duration, "end": 0.94 * duration},
        ]

    for spec in anomalies:
        kind = spec["type"]
        if kind == "disruption_precursor":
            onset = float(spec["onset"])
            growth = float(spec.get("growth_rate", 8.0 / duration))
            mask = t >= onset
            envelope = 0.02 * n0 * np.expm1(growth * (t[mask] - onset))
            x[mask] += envelope * np.sin(2 * np.pi * 3000.0 * t[mask])
            labels[mask] = 1
        elif kind == "sensor_saturation":
            start, end = float(spec["start"]), float(spec["end"])
            mask = (t >= start) & (t <= end)
            cap = float(spec.get("level", 1.02 * n0))
            x[mask] = cap
            labels[mask] = 1
        elif kind == "spike":
            idx = int(np.clip(round(float(spec["time"]) * sample_rate), 0, n - 1))
            x[idx] += float(spec.get("magnitude", 0.5)) * n0
            labels[idx] = 1
        else:
            raise ValueError(f"unknown plasma anomaly type {kind!r}")

    sf = SignalFrame(
        time=t,
        values={"n_e": x},
        name="synthetic_langmuir_probe",
        units={"n_e": "m^-3"},
        sample_rate=sample_rate,
        instrument="langmuir_probe",
        domain="plasma",
        metadata={"synthetic": True, "anomalies": anomalies},
    )
    return sf.add_labels(labels)


def generate_ocean_temperature(
    duration_days: float = 180.0,
    samples_per_day: int = 24,
    anomalies: list[dict] | None = None,
    seed: int | None = 0,
) -> SignalFrame:
    """Moored-buoy sea surface temperature with seasonal and diurnal cycles.

    Injectable anomalies:

    - ``{"type": "sensor_drift", "start_day": d, "rate_per_day": r}`` —
      slow systematic offset growing from ``start_day`` (biofouling-like).
    - ``{"type": "marine_heatwave", "start_day": d0, "end_day": d1,
      "magnitude": m}`` — sustained warm excursion.
    - ``{"type": "spike", "day": d, "magnitude": m}`` — single bad reading.

    If ``anomalies`` is None, one drift and one heatwave are injected.
    """
    rng = np.random.default_rng(seed)
    n = int(duration_days * samples_per_day)
    days = np.arange(n) / samples_per_day
    time = pd.date_range("2025-01-01", periods=n, freq=pd.Timedelta(days=1.0 / samples_per_day))

    x = 15.0 + 4.0 * np.sin(2 * np.pi * days / 365.25 - np.pi / 2)   # seasonal cycle
    x += 0.5 * np.sin(2 * np.pi * days)                              # diurnal cycle
    x += rng.normal(0.0, 0.15, n)                                    # noise

    labels = np.zeros(n, dtype=int)
    if anomalies is None:
        anomalies = [
            {"type": "sensor_drift", "start_day": 0.55 * duration_days, "rate_per_day": 0.04},
            {"type": "marine_heatwave", "start_day": 0.2 * duration_days,
             "end_day": 0.25 * duration_days, "magnitude": 3.0},
        ]

    for spec in anomalies:
        kind = spec["type"]
        if kind == "sensor_drift":
            start = float(spec["start_day"])
            rate = float(spec.get("rate_per_day", 0.04))
            mask = days >= start
            x[mask] += rate * (days[mask] - start)
            labels[mask] = 1
        elif kind == "marine_heatwave":
            d0, d1 = float(spec["start_day"]), float(spec["end_day"])
            mask = (days >= d0) & (days <= d1)
            ramp = np.sin(np.pi * (days[mask] - d0) / max(d1 - d0, 1e-9))
            x[mask] += float(spec.get("magnitude", 3.0)) * ramp
            labels[mask] = 1
        elif kind == "spike":
            idx = int(np.clip(round(float(spec["day"]) * samples_per_day), 0, n - 1))
            x[idx] += float(spec.get("magnitude", 5.0))
            labels[idx] = 1
        else:
            raise ValueError(f"unknown ocean anomaly type {kind!r}")

    sf = SignalFrame(
        time=time,
        values={"temperature": x},
        name="synthetic_ocean_buoy",
        units={"temperature": "°C"},
        instrument="moored_buoy",
        domain="ocean",
        metadata={"synthetic": True, "anomalies": anomalies,
                  "expected_range": {"temperature": (-2.0, 35.0)}},
    )
    return sf.add_labels(labels)


def generate_generic_signal(
    n: int = 2000,
    sample_rate: float = 1.0,
    trend: float = 0.0,
    seasonal_period: int | None = None,
    seasonal_amplitude: float = 1.0,
    noise_std: float = 1.0,
    anomalies: list[dict] | None = None,
    seed: int | None = 0,
) -> SignalFrame:
    """Configurable trend + seasonal + noise signal with injected anomalies.

    Anomaly specs (indices are sample positions):

    - ``{"type": "spike", "index": i, "magnitude": m}``
    - ``{"type": "level_shift", "start": i, "end": j, "magnitude": m}``
    - ``{"type": "noise_burst", "start": i, "end": j, "std": s}``
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n) / sample_rate
    x = trend * t + rng.normal(0.0, noise_std, n)
    if seasonal_period:
        x += seasonal_amplitude * np.sin(2 * np.pi * np.arange(n) / seasonal_period)

    labels = np.zeros(n, dtype=int)
    for spec in anomalies or []:
        kind = spec["type"]
        if kind == "spike":
            i = int(spec["index"])
            x[i] += float(spec.get("magnitude", 8.0 * noise_std))
            labels[i] = 1
        elif kind == "level_shift":
            i, j = int(spec["start"]), int(spec["end"])
            x[i : j + 1] += float(spec.get("magnitude", 5.0 * noise_std))
            labels[i : j + 1] = 1
        elif kind == "noise_burst":
            i, j = int(spec["start"]), int(spec["end"])
            x[i : j + 1] += rng.normal(0.0, float(spec.get("std", 5.0 * noise_std)), j - i + 1)
            labels[i : j + 1] = 1
        else:
            raise ValueError(f"unknown generic anomaly type {kind!r}")

    sf = SignalFrame(
        time=t,
        values={"value": x},
        name="synthetic_generic",
        sample_rate=sample_rate,
        metadata={"synthetic": True},
    )
    return sf.add_labels(labels)
