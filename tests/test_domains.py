import numpy as np
import pytest

from sigmaflow import Pipeline, SignalFrame
from sigmaflow.domains import (
    ADAPTER_REGISTRY,
    CustomAdapter,
    EnergyGridAdapter,
    OceanAdapter,
    PlasmaAdapter,
    SatelliteAdapter,
    get_adapter,
)

ALL_ADAPTERS = [PlasmaAdapter, OceanAdapter, SatelliteAdapter, EnergyGridAdapter]


@pytest.mark.parametrize("cls", ALL_ADAPTERS)
def test_default_pipeline_runs(cls, rng):
    t = np.arange(2000.0)
    y = rng.normal(0, 1, 2000)
    y[1000] = 12.0
    sf = SignalFrame(time=t, values=y, domain=cls.domain)
    adapter = cls()
    pipe = adapter.default_pipeline()
    assert isinstance(pipe, Pipeline)
    result = pipe.fit_detect(sf)
    assert len(result.labels) == len(sf)


@pytest.mark.parametrize("cls", ALL_ADAPTERS)
def test_feature_defaults_are_valid(cls):
    from sigmaflow.features import AVAILABLE_FEATURES

    for feat in cls().feature_defaults():
        assert feat in AVAILABLE_FEATURES


def test_registry_and_get_adapter():
    assert set(ADAPTER_REGISTRY) == {"plasma", "ocean", "satellite", "energy"}
    assert isinstance(get_adapter("ocean"), OceanAdapter)
    with pytest.raises(KeyError):
        get_adapter("astrology")


def test_plasma_validate_range(rng):
    y = rng.normal(1e19, 1e17, 100)
    y[10] = 1e25  # impossible density
    sf = SignalFrame(time=np.arange(100.0), values={"n_e": y},
                     instrument="langmuir_probe", domain="plasma")
    warnings = PlasmaAdapter().validate(sf)
    assert any("outside expected range" in w for w in warnings)


def test_plasma_saturation_filter(rng):
    y = rng.normal(0, 1, 500)
    y[200:220] = y.max() + 5.0  # 20-sample plateau at the ceiling
    sf = SignalFrame(time=np.arange(500.0), values=y)
    out = PlasmaAdapter().known_artifact_filter(sf)
    assert out.values.isna().sum().sum() >= 20
    assert "saturation_flagged" in out.metadata


def test_plasma_classification(rng):
    y = rng.normal(0, 1, 2000)
    y[500] = 15.0            # spike -> rf_interference
    y[1500:] += 4.0          # sustained shift over 25% of record -> drift
    sf = SignalFrame(time=np.arange(2000.0), values=y, domain="plasma")
    from sigmaflow.detectors import ModifiedZScoreDetector

    result = ModifiedZScoreDetector().fit_detect(sf)
    result = PlasmaAdapter().classify_anomaly(result, sf)
    assert len(result.categories) == len(result.intervals)
    assert set(result.categories) <= set(PlasmaAdapter.anomaly_categories) | {"unclassified"}
    assert "rf_interference" in result.categories


def test_ocean_validate_frozen_seawater(rng):
    y = rng.normal(15, 1, 100)
    y[5] = -10.0
    sf = SignalFrame(time=np.arange(100.0), values={"water_temperature": y},
                     instrument="moored_buoy", domain="ocean")
    warnings = OceanAdapter().validate(sf)
    assert any("colder than seawater" in w for w in warnings)


def test_ocean_despike_filter(rng):
    y = rng.normal(15, 0.1, 500)
    y[100] = 40.0
    sf = SignalFrame(time=np.arange(500.0), values={"temperature": y})
    out = OceanAdapter().known_artifact_filter(sf)
    assert abs(out["temperature"].iloc[100] - 15.0) < 1.0  # spike replaced
    assert out.metadata["despiked"]["temperature"] >= 1


def test_energy_frequency_classification(rng):
    nominal = 50.0
    y = rng.normal(nominal, 0.02, 3000)
    y[2000:2400] -= 0.4  # sustained under-frequency: generation loss
    sf = SignalFrame(time=np.arange(3000.0), values={"frequency": y},
                     domain="energy", metadata={"nominal_frequency": nominal})
    from sigmaflow.detectors import ModifiedZScoreDetector

    result = ModifiedZScoreDetector(threshold=5.0).fit_detect(sf)
    result = EnergyGridAdapter().classify_anomaly(result, sf)
    assert "generation_dropout" in result.categories


def test_energy_validate_excursion(rng):
    y = rng.normal(50.0, 0.02, 100)
    y[50] = 50.8
    sf = SignalFrame(time=np.arange(100.0), values={"frequency": y},
                     metadata={"nominal_frequency": 50.0})
    warnings = EnergyGridAdapter().validate(sf)
    assert any("operating-band" in w for w in warnings)


def test_missing_data_warning():
    y = np.full(100, np.nan)
    y[:50] = 1.0
    sf = SignalFrame(time=np.arange(100.0), values=y)
    warnings = SatelliteAdapter().validate(sf)
    assert any("missing" in w for w in warnings)


def test_custom_adapter():
    adapter = CustomAdapter(
        domain="seismology",
        instrument_profiles={"seismometer": {"expected_range": {"v": (-1, 1)}}},
        anomaly_categories=["earthquake", "sensor_glitch"],
        default_features=["kurtosis", "spectral_entropy"],
        category_map={"spike": "sensor_glitch", "noisy": "earthquake",
                      "shift": "earthquake", "flat": "sensor_glitch"},
    )
    assert adapter.domain == "seismology"
    assert adapter.feature_defaults() == ["kurtosis", "spectral_entropy"]

    rng = np.random.default_rng(0)
    y = rng.normal(0, 1e-4, 1000)
    y[500] = 0.5
    sf = SignalFrame(time=np.arange(1000.0), values={"v": y},
                     instrument="seismometer")
    from sigmaflow.detectors import ModifiedZScoreDetector

    result = ModifiedZScoreDetector().fit_detect(sf)
    result = adapter.classify_anomaly(result, sf)
    assert "sensor_glitch" in result.categories
    warnings = adapter.validate(sf)
    assert len(warnings) >= 0  # runs cleanly with a profile
