import numpy as np
import pytest

from sigmaflow import Pipeline, SignalFrame
from sigmaflow.detectors import IsolationForestDetector, ZScoreDetector
from sigmaflow.preprocess import Detrend, GapHandler, Normalizer, Resampler


def make_pipeline():
    return Pipeline([
        GapHandler(max_gap="5s", fill_method="interpolate"),
        Detrend(method="linear"),
        Normalizer(method="robust"),
        ZScoreDetector(threshold=3.5),
    ])


def test_fit_detect(simple_signal):
    result = make_pipeline().fit_detect(simple_signal)
    assert len(result.labels) == len(simple_signal)
    assert result.labels[500] == 1


def test_transform_only(simple_signal):
    pipe = Pipeline([Detrend(method="linear"), Normalizer(method="z_score")])
    out = pipe.fit(simple_signal).transform(simple_signal)
    assert isinstance(out, SignalFrame)
    assert out["value"].std(ddof=0) == pytest.approx(1.0, abs=1e-6)


def test_detect_without_detector_raises(simple_signal):
    pipe = Pipeline([Detrend()])
    with pytest.raises(ValueError, match="no detector"):
        pipe.detect(simple_signal)


def test_detector_must_be_last():
    with pytest.raises(ValueError, match="final"):
        Pipeline([ZScoreDetector(), Detrend()])


def test_empty_pipeline_raises():
    with pytest.raises(ValueError):
        Pipeline([])


def test_non_component_raises():
    with pytest.raises(TypeError):
        Pipeline(["not a step"])
    with pytest.raises(TypeError):
        Pipeline([Detrend(), object()])


def test_yaml_roundtrip(tmp_path, simple_signal):
    pipe = make_pipeline()
    path = str(tmp_path / "pipe.yaml")
    pipe.save(path)
    loaded = Pipeline.load(path)
    assert [type(s) for s in loaded.steps] == [type(s) for s in pipe.steps]
    assert loaded.steps[-1].threshold == 3.5
    r1 = pipe.fit_detect(simple_signal)
    r2 = loaded.fit_detect(simple_signal)
    np.testing.assert_array_equal(r1.labels, r2.labels)


def test_yaml_roundtrip_with_tuple_param(tmp_path):
    pipe = Pipeline([Normalizer(method="z_score", fit_on=(0.0, 100.0)), ZScoreDetector()])
    path = str(tmp_path / "pipe.yaml")
    pipe.save(path)
    loaded = Pipeline.load(path)
    assert loaded.steps[0].fit_on == (0.0, 100.0)


def test_yaml_with_ml_detector(tmp_path, simple_signal):
    pipe = Pipeline([Resampler(target_rate=0.5), IsolationForestDetector(contamination=0.05)])
    path = str(tmp_path / "pipe.yaml")
    pipe.save(path)
    result = Pipeline.load(path).fit_detect(simple_signal)
    assert len(result.labels) == pytest.approx(len(simple_signal) / 2, abs=2)


def test_load_rejects_garbage(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("just: nonsense")
    with pytest.raises(ValueError, match="not a sigmaflow pipeline"):
        Pipeline.load(str(path))


def test_load_rejects_unknown_component(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("sigmaflow_pipeline: 1\nsteps:\n- class: warp_drive\n  params: {}\n")
    with pytest.raises(ValueError, match="unknown pipeline component"):
        Pipeline.load(str(path))


def test_repr():
    assert "ZScoreDetector" in repr(make_pipeline())
