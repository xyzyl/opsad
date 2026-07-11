import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="deep extra not installed")

from sigmaflow import SignalFrame  # noqa: E402
from sigmaflow.detectors import (  # noqa: E402
    DETECTOR_REGISTRY,
    AutoencoderDetector,
    LSTMAutoencoderDetector,
    TransformerDetector,
)

FAST = dict(window_size=16, epochs=4, batch_size=32)


@pytest.fixture
def wave_with_glitch(rng):
    """A clean periodic signal with a burst the model has never seen."""
    t = np.arange(1200.0)
    y = np.sin(2 * np.pi * t / 40) + rng.normal(0, 0.05, 1200)
    y[800:820] += rng.normal(0, 1.5, 20)
    return SignalFrame(time=t, values=y, name="wave")


@pytest.mark.parametrize("cls,kwargs", [
    (AutoencoderDetector, dict(architecture="fc", **FAST)),
    (AutoencoderDetector, dict(architecture="conv1d", **FAST)),
    (LSTMAutoencoderDetector, dict(lstm_hidden_size=16, **FAST)),
    (TransformerDetector, dict(d_model=16, n_heads=2, n_layers=1, **FAST)),
])
def test_reconstruction_detectors_find_burst(cls, kwargs, wave_with_glitch):
    sig = wave_with_glitch
    clean = sig.slice(0.0, 700.0)
    det = cls(**kwargs)
    result = det.fit(clean).detect(sig)
    assert len(result.labels) == len(sig)
    assert np.isfinite(result.scores).all()
    # the unseen burst must score far above the normal region
    burst = result.scores[800:820].mean()
    normal = result.scores[100:700].mean()
    assert burst > 3 * normal


def test_registered_in_registry():
    for name in ("autoencoder", "lstm_autoencoder", "transformer"):
        assert name in DETECTOR_REGISTRY


def test_reproducible(wave_with_glitch):
    a = AutoencoderDetector(random_state=1, **FAST).fit_detect(wave_with_glitch)
    b = AutoencoderDetector(random_state=1, **FAST).fit_detect(wave_with_glitch)
    np.testing.assert_allclose(a.scores, b.scores, rtol=1e-4)


def test_bad_architecture():
    with pytest.raises(ValueError):
        AutoencoderDetector(architecture="quantum")


def test_short_signal(rng):
    sf = SignalFrame(time=np.arange(30.0), values=rng.normal(0, 1, 30))
    det = AutoencoderDetector(**FAST)
    result = det.fit_detect(sf)
    assert len(result.labels) == 30


def test_params_roundtrip():
    det = TransformerDetector(d_model=32, n_heads=2, **FAST)
    params = det.get_params()
    assert params["d_model"] == 32
    assert params["window_size"] == 16
