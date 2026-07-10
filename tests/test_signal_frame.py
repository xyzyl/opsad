import numpy as np
import pandas as pd
import pytest

from sigmaflow import SignalFrame


def test_minimal_creation():
    sf = SignalFrame(time=[0.0, 1.0, 2.0], values=[1.0, 2.0, 3.0])
    assert sf.channels == ["value"]
    assert len(sf) == 3
    assert sf.name is None
    assert sf.units == {} and sf.metadata == {}


def test_dict_creation_with_metadata():
    sf = SignalFrame(
        time=[0.0, 0.5, 1.0],
        values={"n_e": [1e19, 2e19, 3e19], "T_e": [10.0, 20.0, 30.0]},
        name="probe",
        units={"n_e": "m^-3", "T_e": "eV"},
        sample_rate=2.0,
        instrument="langmuir_probe",
        domain="plasma",
        metadata={"shot_number": 184520},
    )
    assert sf.channels == ["n_e", "T_e"]
    assert sf.units["T_e"] == "eV"
    assert sf.sample_rate == 2.0
    assert sf.domain == "plasma"
    assert sf.metadata["shot_number"] == 184520
    assert list(sf["T_e"]) == [10.0, 20.0, 30.0]


def test_2d_array_creation():
    sf = SignalFrame(time=np.arange(4.0), values=np.ones((4, 2)))
    assert sf.channels == ["channel_0", "channel_1"]


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="samples"):
        SignalFrame(time=[0.0, 1.0], values=[1.0, 2.0, 3.0])


def test_empty_time_raises():
    with pytest.raises(ValueError):
        SignalFrame(time=[], values=[])


def test_sample_rate_inferred():
    sf = SignalFrame(time=np.arange(0, 10, 0.25), values=np.zeros(40))
    assert sf.sample_rate == pytest.approx(4.0)


def test_duration_float_and_datetime():
    sf = SignalFrame(time=np.arange(0.0, 5.0, 0.5), values=np.zeros(10))
    assert sf.duration == pytest.approx(4.5)
    time = pd.date_range("2025-01-01", periods=61, freq="1s")
    sf2 = SignalFrame(time=time, values=np.zeros(61))
    assert sf2.duration == pytest.approx(60.0)
    assert sf2.sample_rate == pytest.approx(1.0)


def test_slice(simple_signal):
    part = simple_signal.slice(100.0, 199.0)
    assert len(part) == 100
    assert part.time[0] == 100.0


def test_slice_datetime(datetime_signal):
    part = datetime_signal.slice("2025-06-01 00:10:00", "2025-06-01 00:19:00")
    assert len(part) == 10


def test_dropna_and_interpolate():
    y = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
    sf = SignalFrame(time=np.arange(5.0), values=y)
    assert len(sf.dropna()) == 3
    filled = sf.interpolate("linear")
    assert not filled.values.isna().any().any()
    assert filled["value"].iloc[1] == pytest.approx(2.0)


def test_interpolate_bad_method(simple_signal):
    with pytest.raises(ValueError):
        simple_signal.interpolate("quintic")


def test_resample_upsample(simple_signal):
    up = simple_signal.resample(2.0)
    assert up.sample_rate == pytest.approx(2.0)
    assert len(up) == pytest.approx(2 * len(simple_signal), abs=2)


def test_resample_invalid(simple_signal):
    with pytest.raises(ValueError):
        simple_signal.resample(0)


def test_add_labels_roundtrip(simple_signal):
    labels = np.zeros(len(simple_signal), dtype=int)
    labels[500] = 1
    out = simple_signal.add_labels(labels)
    assert out is simple_signal
    assert simple_signal.anomaly_labels.sum() == 1
    with pytest.raises(ValueError):
        simple_signal.add_labels([0, 1])


def test_metadata_preserved_through_slice(multichannel_signal):
    part = multichannel_signal.slice(0.0, 99.0)
    assert part.units == multichannel_signal.units
    assert part.metadata == multichannel_signal.metadata
    assert part.name == multichannel_signal.name


def test_to_numpy_and_dataframe(multichannel_signal):
    arr = multichannel_signal.to_numpy()
    assert arr.shape == (500, 2)
    df = multichannel_signal.to_dataframe()
    assert list(df.columns) == ["a", "b"]


def test_hdf5_roundtrip(tmp_path, multichannel_signal):
    labels = np.zeros(len(multichannel_signal), dtype=int)
    labels[250] = 1
    multichannel_signal.add_labels(labels)
    path = str(tmp_path / "sig.h5")
    multichannel_signal.to_hdf5(path)
    back = SignalFrame.from_hdf5(path)
    assert back.channels == multichannel_signal.channels
    assert back.units == multichannel_signal.units
    assert back.metadata["origin"] == "test"
    np.testing.assert_allclose(back.to_numpy(), multichannel_signal.to_numpy())
    assert back.anomaly_labels.sum() == 1


def test_hdf5_roundtrip_datetime(tmp_path, datetime_signal):
    path = str(tmp_path / "dt.h5")
    datetime_signal.to_hdf5(path)
    back = SignalFrame.from_hdf5(path)
    assert isinstance(back.time, pd.DatetimeIndex)
    assert back.time[0] == datetime_signal.time[0]


def test_gap_report():
    t = np.concatenate([np.arange(0, 10.0), np.arange(20.0, 30.0)])
    sf = SignalFrame(time=t, values=np.zeros(20))
    gaps = sf.gap_report()
    assert len(gaps) == 1
    assert gaps[0][2] == pytest.approx(11.0)  # from t=9 to t=20


def test_summary_runs(multichannel_signal, capsys):
    text = multichannel_signal.summary()
    assert "multi" in text
    assert "a [V]" in text
    assert capsys.readouterr().out  # printed


def test_repr(simple_signal):
    assert "simple" in repr(simple_signal)
