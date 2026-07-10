import matplotlib

matplotlib.use("Agg")  # headless plotting in tests

import numpy as np
import pandas as pd
import pytest

from sigmaflow import SignalFrame


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def simple_signal(rng):
    """1000 samples of unit-variance noise with a big spike at 500."""
    t = np.arange(1000, dtype=float)
    y = rng.normal(0, 1, 1000)
    y[500] = 15.0
    return SignalFrame(time=t, values=y, name="simple")


@pytest.fixture
def shifted_signal(rng):
    """Mean shift of +3 sigma starting at sample 600."""
    t = np.arange(1000, dtype=float)
    y = rng.normal(0, 1, 1000)
    y[600:] += 3.0
    return SignalFrame(time=t, values=y, name="shifted")


@pytest.fixture
def seasonal_signal(rng):
    """Strong 50-sample seasonality with one off-cycle spike at 375."""
    n = 1000
    t = np.arange(n, dtype=float)
    y = 10.0 * np.sin(2 * np.pi * t / 50) + rng.normal(0, 0.5, n)
    y[375] += 8.0
    return SignalFrame(time=t, values=y, name="seasonal")


@pytest.fixture
def multichannel_signal(rng):
    t = np.arange(500, dtype=float)
    a = rng.normal(0, 1, 500)
    b = rng.normal(5, 2, 500)
    b[250] = 50.0
    return SignalFrame(
        time=t,
        values={"a": a, "b": b},
        name="multi",
        units={"a": "V", "b": "K"},
        metadata={"origin": "test"},
    )


@pytest.fixture
def datetime_signal(rng):
    time = pd.date_range("2025-06-01", periods=300, freq="1min")
    y = rng.normal(20, 0.5, 300)
    return SignalFrame(time=time, values={"temperature": y}, name="dt", domain="ocean")
