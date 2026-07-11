import numpy as np
import pandas as pd
import pytest

from sigmaflow.data import sources as src
from sigmaflow.data.sources import (
    LIVE_SOURCES,
    SourceUnavailable,
    fetch_all_live,
    fetch_gb_grid_frequency,
    fetch_ndbc_buoy,
    parse_elexon_frequency,
    parse_goes_mag,
    parse_ndbc_realtime2,
    parse_swpc_rtsw,
)

NDBC_SAMPLE = """#YY  MM DD hh mm WDIR WSPD GST  WVHT   DPD   APD MWD   PRES  ATMP  WTMP  DEWP  VIS PTDY  TIDE
#yr  mo dy hr mn degT m/s  m/s     m   sec   sec degT   hPa  degC  degC  degC  nmi  hPa    ft
2026 07 10 20 40 320 10.0 12.0    MM    MM    MM  MM 1014.4  13.3  15.3  11.6   MM   MM    MM
2026 07 10 20 30 320 10.0 12.0    MM    MM    MM  MM 1014.5  13.2  15.3  11.4   MM   MM    MM
2026 07 10 20 20 320  9.0 11.0    MM    MM    MM  MM 1014.6  13.1  15.2  11.4   MM   MM    MM
2026 07 10 20 10 310  9.0 11.0    MM    MM    MM  MM   MM     MM    MM   11.3   MM   MM    MM
"""


def test_parse_ndbc():
    times, data = parse_ndbc_realtime2(
        NDBC_SAMPLE, {"WTMP": "water_temperature", "PRES": "pressure"}
    )
    # newest-first input becomes oldest-first output; all-MM row dropped
    assert len(times) == 3
    assert times[0] < times[-1]
    assert data["water_temperature"] == [15.2, 15.3, 15.3]
    assert data["pressure"][0] == 1014.6


def test_parse_ndbc_bad_format():
    with pytest.raises(SourceUnavailable):
        parse_ndbc_realtime2("<html>error</html>", {"WTMP": "x"})
    with pytest.raises(SourceUnavailable, match="columns not found"):
        parse_ndbc_realtime2(NDBC_SAMPLE, {"NOPE": "x"})


def test_parse_swpc_rtsw():
    payload = [
        {"time_tag": "2026-07-10T21:11:00", "proton_speed": 600.1,
         "proton_temperature": 208760, "proton_density": 2.98},
        {"time_tag": "2026-07-10T21:12:00", "proton_speed": None,
         "proton_temperature": 209000, "proton_density": 3.01},
        {"time_tag": "bogus"},
    ]
    times, data = parse_swpc_rtsw(payload)
    assert len(times) == 2
    assert data["proton_speed"][0] == 600.1
    assert np.isnan(data["proton_speed"][1])


def test_parse_goes_mag():
    payload = [
        {"time_tag": "2026-07-09T21:16:00Z", "He": 15.5, "Hp": 111.1,
         "Hn": 12.3, "total": 112.8, "arcjet_flag": False},
        {"time_tag": "2026-07-09T21:17:00Z", "He": 15.9, "Hp": 111.0,
         "Hn": None, "total": 112.7},
    ]
    times, data = parse_goes_mag(payload)
    assert len(times) == 2
    assert data["Hp"] == [111.1, 111.0]
    assert np.isnan(data["Hn"][1])


def test_parse_elexon_sorts_ascending():
    payload = {"data": [
        {"measurementTime": "2026-07-09T00:00:30Z", "frequency": 50.017},
        {"measurementTime": "2026-07-09T00:00:00Z", "frequency": 50.031},
        {"measurementTime": "2026-07-09T00:00:15Z", "frequency": 50.030},
        {"measurementTime": None, "frequency": 50.0},
    ]}
    times, data = parse_elexon_frequency(payload)
    assert len(times) == 3
    assert times == sorted(times)
    assert data["frequency"] == [50.031, 50.030, 50.017]


def test_fetch_ndbc_via_mock(monkeypatch):
    # 12 valid rows so the >=10 sample guard passes
    rows = "\n".join(
        f"2026 07 10 {20 - h:02d} 00 320 10.0 12.0 MM MM MM MM 1014.{h % 10} "
        f"13.{h % 10} 15.{h % 10} 11.0 MM MM MM"
        for h in range(12)
    )
    text = NDBC_SAMPLE.splitlines()[0] + "\n" + NDBC_SAMPLE.splitlines()[1] + "\n" + rows
    monkeypatch.setattr(src, "_http_get", lambda url, timeout=30.0: text)
    sf = fetch_ndbc_buoy("46042")
    assert sf.domain == "ocean"
    assert sf.units["water_temperature"] == "°C"
    assert isinstance(sf.time, pd.DatetimeIndex)
    assert sf.metadata["attribution"].startswith("Data: NOAA")
    assert "fetched_at" in sf.metadata


def test_fetch_too_few_samples(monkeypatch):
    monkeypatch.setattr(src, "_http_get", lambda url, timeout=30.0: NDBC_SAMPLE)
    with pytest.raises(SourceUnavailable, match="usable samples"):
        fetch_ndbc_buoy("46042")


def test_fetch_gb_grid_frequency_via_mock(monkeypatch):
    import json as _json

    data = {"data": [
        {"measurementTime": f"2026-07-09T00:{m:02d}:00Z", "frequency": 50.0 + m / 1000}
        for m in range(15)
    ]}
    monkeypatch.setattr(src, "_http_get", lambda url, timeout=30.0: _json.dumps(data))
    sf = fetch_gb_grid_frequency(hours=1)
    assert sf.domain == "energy"
    assert sf.metadata["nominal_frequency"] == 50.0
    assert len(sf) == 15


def test_fetch_all_live_skips_failures(monkeypatch):
    def boom(timeout=30.0):
        raise SourceUnavailable("down")

    ok_signal = fetch_stub()
    monkeypatch.setitem(LIVE_SOURCES, "plasma: solar wind at L1 (live)", boom)
    monkeypatch.setitem(LIVE_SOURCES, "ocean: NDBC buoy 46042 (live)",
                        lambda timeout=30.0: ok_signal)
    monkeypatch.setitem(LIVE_SOURCES, "satellite: GOES magnetometer (live)", boom)
    monkeypatch.setitem(LIVE_SOURCES, "energy: GB grid frequency (live)", boom)
    out = fetch_all_live()
    assert list(out) == ["ocean: NDBC buoy 46042 (live)"]
    with pytest.raises(SourceUnavailable):
        fetch_all_live(strict=True)


def fetch_stub():
    from sigmaflow import SignalFrame

    return SignalFrame(time=np.arange(20.0), values=np.zeros(20))


def test_http_get_bad_url():
    with pytest.raises(SourceUnavailable):
        src._http_get("http://127.0.0.1:1/nothing", timeout=2)
