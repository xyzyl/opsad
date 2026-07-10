import numpy as np
import pandas as pd
import pytest

from sigmaflow import SignalFrame
from sigmaflow.preprocess import Detrend, GapHandler, Normalizer, Resampler


# ---------------------------------------------------------------- Resampler

def test_resampler_downsample_mean():
    t = np.arange(0, 100, 0.1)  # 10 Hz
    sf = SignalFrame(time=t, values=np.sin(t))
    out = Resampler(target_rate=1.0).fit_transform(sf)
    assert out.sample_rate == pytest.approx(1.0)
    assert len(out) == pytest.approx(100, abs=2)
    assert not out.values.isna().any().any()


def test_resampler_downsample_methods():
    t = np.arange(0, 50, 0.1)
    y = np.tile([0.0, 10.0], len(t) // 2)
    sf = SignalFrame(time=t, values=y)
    hi = Resampler(target_rate=1.0, method="max", anti_alias=False).fit_transform(sf)
    lo = Resampler(target_rate=1.0, method="min", anti_alias=False).fit_transform(sf)
    assert np.all(hi.to_numpy() >= lo.to_numpy())
    assert hi["value"].max() == pytest.approx(10.0)
    assert lo["value"].min() == pytest.approx(0.0)


def test_resampler_upsample_linear_and_cubic():
    t = np.arange(0, 10.0)
    sf = SignalFrame(time=t, values=t * 2.0)
    for method in ("linear", "cubic"):
        up = Resampler(target_rate=4.0, method=method).fit_transform(sf)
        assert up.sample_rate == pytest.approx(4.0)
        # a straight line must be reproduced exactly by both interpolants
        np.testing.assert_allclose(up["value"].to_numpy()[:-1],
                                   2.0 * np.asarray(up.time)[:-1], atol=1e-9)


def test_resampler_invalid():
    with pytest.raises(ValueError):
        Resampler(target_rate=-1)
    with pytest.raises(ValueError):
        Resampler(target_rate=1.0, method="bogus")
    t = np.arange(0, 100, 0.1)
    sf = SignalFrame(time=t, values=np.zeros(len(t)))
    with pytest.raises(ValueError, match="interpolation"):
        Resampler(target_rate=1.0, method="cubic").fit_transform(sf)


def test_resampler_preserves_metadata():
    t = np.arange(0, 100, 0.1)
    sf = SignalFrame(time=t, values=np.sin(t), units={"value": "V"}, domain="energy")
    out = Resampler(target_rate=1.0).fit_transform(sf)
    assert out.units == {"value": "V"}
    assert out.domain == "energy"


# ---------------------------------------------------------------- GapHandler

def test_gap_handler_fills_small_gap():
    t = np.concatenate([np.arange(0, 10.0), np.arange(13.0, 23.0)])  # 3 s gap
    sf = SignalFrame(time=t, values=t.copy())
    out = GapHandler(max_gap="5s", fill_method="interpolate").fit_transform(sf)
    assert len(out) > len(sf)
    secs = np.asarray(out.time, dtype=float)
    assert np.max(np.diff(secs)) < 1.5  # gap gone
    np.testing.assert_allclose(out["value"].to_numpy(), secs, atol=1e-9)  # linear fill exact


def test_gap_handler_reports_large_gap():
    t = np.concatenate([np.arange(0, 10.0), np.arange(100.0, 110.0)])
    sf = SignalFrame(time=t, values=np.zeros(20))
    out = GapHandler(max_gap="5s").fit_transform(sf)
    assert len(out) == len(sf)  # too large to fill
    assert len(out.metadata["gaps"]) == 1
    assert out.metadata["gaps"][0]["duration_s"] == pytest.approx(91.0)  # t=9 to t=100


def test_gap_handler_fill_methods():
    t = np.concatenate([np.arange(0, 5.0), np.arange(7.0, 12.0)])
    y = np.concatenate([np.full(5, 1.0), np.full(5, 9.0)])
    sf = SignalFrame(time=t, values=y)
    zero = GapHandler(max_gap=5.0, fill_method="zero").fit_transform(sf)
    assert (zero["value"] == 0.0).sum() >= 1
    ffill = GapHandler(max_gap=5.0, fill_method="forward_fill").fit_transform(sf)
    filled_vals = ffill["value"].to_numpy()
    assert set(np.unique(filled_vals)) == {1.0, 9.0}
    nan = GapHandler(max_gap=5.0, fill_method="nan").fit_transform(sf)
    assert nan.values.isna().any().any()


def test_gap_handler_invalid_method():
    with pytest.raises(ValueError):
        GapHandler(fill_method="magic")


def test_gap_handler_datetime_index():
    time = pd.date_range("2025-01-01", periods=10, freq="1s")
    time = time.delete([4, 5])  # 3 s gap
    sf = SignalFrame(time=time, values=np.arange(8.0))
    out = GapHandler(max_gap="5s").fit_transform(sf)
    assert len(out) == 10
    assert isinstance(out.time, pd.DatetimeIndex)


# ---------------------------------------------------------------- Detrend

def test_detrend_linear_removes_slope(rng):
    t = np.arange(500.0)
    y = 0.05 * t + rng.normal(0, 0.1, 500)
    sf = SignalFrame(time=t, values=y)
    out = Detrend(method="linear").fit_transform(sf)
    assert abs(np.polyfit(t, out["value"], 1)[0]) < 1e-6


def test_detrend_polynomial():
    t = np.arange(200.0)
    y = 0.001 * t**2 - 0.1 * t + 5
    sf = SignalFrame(time=t, values=y)
    out = Detrend(method="polynomial", order=2).fit_transform(sf)
    np.testing.assert_allclose(out["value"].to_numpy(), 0.0, atol=1e-6)


def test_detrend_moving_average_and_inverse(rng):
    t = np.arange(300.0)
    y = np.sin(t / 20) + rng.normal(0, 0.05, 300)
    sf = SignalFrame(time=t, values=y)
    d = Detrend(method="moving_average", window=51)
    out = d.fit_transform(sf)
    restored = d.inverse_transform(out)
    np.testing.assert_allclose(restored["value"].to_numpy(), y, atol=1e-9)


def test_detrend_differencing_and_inverse():
    t = np.arange(100.0)
    y = np.cumsum(np.ones(100))
    sf = SignalFrame(time=t, values=y)
    d = Detrend(method="differencing")
    out = d.fit_transform(sf)
    assert out["value"].iloc[0] == 0.0
    np.testing.assert_allclose(out["value"].to_numpy()[1:], 1.0)
    restored = d.inverse_transform(out)
    np.testing.assert_allclose(restored["value"].to_numpy(), y)


def test_detrend_inverse_without_transform_raises(simple_signal):
    with pytest.raises(RuntimeError):
        Detrend(method="linear").inverse_transform(simple_signal)


def test_detrend_invalid_method():
    with pytest.raises(ValueError):
        Detrend(method="fourier")


# ---------------------------------------------------------------- Normalizer

def test_normalizer_zscore(simple_signal):
    out = Normalizer(method="z_score").fit_transform(simple_signal)
    assert out["value"].mean() == pytest.approx(0.0, abs=1e-9)
    assert out["value"].std(ddof=0) == pytest.approx(1.0, abs=1e-9)


def test_normalizer_minmax(simple_signal):
    out = Normalizer(method="min_max").fit_transform(simple_signal)
    assert out["value"].min() == pytest.approx(0.0)
    assert out["value"].max() == pytest.approx(1.0)


def test_normalizer_robust_ignores_outlier(rng):
    y = rng.normal(100, 5, 1000)
    y[0] = 1e6
    sf = SignalFrame(time=np.arange(1000.0), values=y)
    out = Normalizer(method="robust").fit_transform(sf)
    assert abs(out["value"].iloc[1:].median()) < 0.1


def test_normalizer_inverse_roundtrip(multichannel_signal):
    norm = Normalizer(method="z_score")
    out = norm.fit_transform(multichannel_signal)
    back = norm.inverse_transform(out)
    np.testing.assert_allclose(back.to_numpy(), multichannel_signal.to_numpy(), atol=1e-9)


def test_normalizer_fit_on_interval(shifted_signal):
    norm = Normalizer(method="z_score", fit_on=(0.0, 599.0))
    out = norm.fit(shifted_signal).transform(shifted_signal)
    # normal region ~N(0,1) after scaling; shifted region visibly offset
    assert abs(out["value"].iloc[:600].mean()) < 0.2
    assert out["value"].iloc[600:].mean() > 2.0


def test_normalizer_unfitted_channel_raises(simple_signal, multichannel_signal):
    norm = Normalizer().fit(simple_signal)
    with pytest.raises(KeyError):
        norm.transform(multichannel_signal)


def test_normalizer_inverse_before_fit_raises(simple_signal):
    with pytest.raises(RuntimeError):
        Normalizer().inverse_transform(simple_signal)


def test_normalizer_invalid_method():
    with pytest.raises(ValueError):
        Normalizer(method="quantile")
