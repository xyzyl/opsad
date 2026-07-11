"""Live open-data sources: real instrument streams, no API keys.

Each fetcher returns a :class:`SignalFrame` carrying units, domain, and
attribution metadata. Parsers are separated from fetchers so they can be
tested offline.

Sources (all public, no authentication):

- **NOAA NDBC** moored buoys (ocean): water/air temperature, pressure.
- **NOAA SWPC** real-time solar wind (plasma): proton density, speed,
  temperature measured upstream of Earth (DSCOVR/ACE).
- **NOAA SWPC / GOES** magnetometer (satellite): magnetic field at
  geostationary orbit.
- **Elexon Insights** (energy): Great Britain grid frequency at 15 s.
"""

from __future__ import annotations

import json
import urllib.request
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from ..core.signal_frame import SignalFrame

__all__ = [
    "SourceUnavailable",
    "fetch_ndbc_buoy",
    "fetch_solar_wind",
    "fetch_goes_magnetometer",
    "fetch_gb_grid_frequency",
    "LIVE_SOURCES",
    "fetch_all_live",
]

_USER_AGENT = "sigmaflow/0.2 (+https://github.com/iskander/sigmaflow)"


class SourceUnavailable(RuntimeError):
    """Raised when a live data source can't be reached or parsed."""


def _http_get(url: str, timeout: float = 30.0) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except Exception as exc:
        raise SourceUnavailable(f"could not fetch {url}: {exc}") from exc


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _finish(time, values, **kwargs) -> SignalFrame:
    sf = SignalFrame(time=time, values=values, **kwargs)
    if len(sf) < 10:
        raise SourceUnavailable(
            f"source returned only {len(sf)} usable samples for {kwargs.get('name')}"
        )
    return sf


# ------------------------------------------------------------------ NDBC

def parse_ndbc_realtime2(text: str, channels: dict[str, str]) -> tuple[list, dict]:
    """Parse an NDBC realtime2 text table (newest row first, 'MM' = missing).

    ``channels`` maps NDBC column names to output channel names,
    e.g. ``{"WTMP": "water_temperature"}``.
    """
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 3 or not lines[0].startswith("#"):
        raise SourceUnavailable("unexpected NDBC response format")
    header = lines[0].lstrip("#").split()
    col = {name: i for i, name in enumerate(header)}
    missing = [c for c in channels if c not in col]
    if missing:
        raise SourceUnavailable(f"NDBC columns not found: {missing}")

    times, data = [], {out: [] for out in channels.values()}
    for ln in lines[2:]:
        parts = ln.split()
        if len(parts) < len(header):
            continue
        try:
            t = datetime(int(parts[0]), int(parts[1]), int(parts[2]),
                         int(parts[3]), int(parts[4]), tzinfo=timezone.utc)
        except ValueError:
            continue
        row = {}
        for ndbc_name, out in channels.items():
            raw = parts[col[ndbc_name]]
            row[out] = np.nan if raw == "MM" else float(raw)
        if all(np.isnan(v) for v in row.values()):
            continue
        times.append(t)
        for out, v in row.items():
            data[out].append(v)

    times = times[::-1]  # NDBC lists newest first
    data = {k: v[::-1] for k, v in data.items()}
    return times, data


def fetch_ndbc_buoy(station: str = "46042", timeout: float = 30.0) -> SignalFrame:
    """Last ~45 days of moored-buoy measurements from NOAA NDBC.

    Default station 46042 is the Monterey Bay buoy. Channels: water
    temperature, air temperature, and sea-level pressure.
    """
    url = f"https://www.ndbc.noaa.gov/data/realtime2/{station}.txt"
    text = _http_get(url, timeout)
    channels = {"WTMP": "water_temperature", "ATMP": "air_temperature",
                "PRES": "pressure"}
    times, data = parse_ndbc_realtime2(text, channels)
    return _finish(
        time=pd.DatetimeIndex(times).tz_localize(None),
        values=data,
        name=f"NDBC buoy {station} (live)",
        units={"water_temperature": "°C", "air_temperature": "°C", "pressure": "hPa"},
        instrument="moored_buoy",
        domain="ocean",
        metadata={
            "source": "NOAA National Data Buoy Center",
            "station": station,
            "url": url,
            "attribution": "Data: NOAA NDBC (public domain)",
            "fetched_at": _now_iso(),
        },
    )


# ------------------------------------------------------------------ SWPC solar wind

def parse_swpc_rtsw(payload: list[dict]) -> tuple[list, dict]:
    """Parse the SWPC real-time solar wind 1-minute JSON feed."""
    times, data = [], {"proton_density": [], "proton_speed": [], "proton_temperature": []}
    for rec in payload:
        try:
            t = datetime.fromisoformat(rec["time_tag"].replace("Z", ""))
        except (KeyError, ValueError):
            continue
        vals = {k: rec.get(k) for k in data}
        if all(v is None for v in vals.values()):
            continue
        times.append(t)
        for k, v in vals.items():
            data[k].append(np.nan if v is None else float(v))
    return times, data


def fetch_solar_wind(timeout: float = 30.0) -> SignalFrame:
    """Real-time solar wind plasma (proton density, speed, temperature)
    measured ~1.5 million km upstream of Earth (NOAA SWPC RTSW feed)."""
    url = "https://services.swpc.noaa.gov/json/rtsw/rtsw_wind_1m.json"
    payload = json.loads(_http_get(url, timeout))
    times, data = parse_swpc_rtsw(payload)
    if times and times[0] > times[-1]:
        times = times[::-1]
        data = {k: v[::-1] for k, v in data.items()}
    return _finish(
        time=pd.DatetimeIndex(times),
        values=data,
        name="solar wind plasma at L1 (live)",
        units={"proton_density": "cm^-3", "proton_speed": "km/s",
               "proton_temperature": "K"},
        instrument="solar_wind_plasma_analyzer",
        domain="plasma",
        metadata={
            "source": "NOAA Space Weather Prediction Center (real-time solar wind)",
            "url": url,
            "attribution": "Data: NOAA SWPC / DSCOVR-ACE (public domain)",
            "fetched_at": _now_iso(),
        },
    )


# ------------------------------------------------------------------ GOES magnetometer

def parse_goes_mag(payload: list[dict]) -> tuple[list, dict]:
    """Parse the GOES primary magnetometer 1-minute JSON feed."""
    times, data = [], {"Hp": [], "He": [], "Hn": [], "total": []}
    for rec in payload:
        try:
            t = datetime.fromisoformat(rec["time_tag"].replace("Z", ""))
        except (KeyError, ValueError):
            continue
        times.append(t)
        for k in data:
            v = rec.get(k)
            data[k].append(np.nan if v is None else float(v))
    return times, data


def fetch_goes_magnetometer(timeout: float = 30.0) -> SignalFrame:
    """Magnetic field measured by the primary GOES satellite in
    geostationary orbit over the last day (NOAA SWPC feed)."""
    url = "https://services.swpc.noaa.gov/json/goes/primary/magnetometers-1-day.json"
    payload = json.loads(_http_get(url, timeout))
    times, data = parse_goes_mag(payload)
    return _finish(
        time=pd.DatetimeIndex(times),
        values=data,
        name="GOES magnetometer (live)",
        units={"Hp": "nT", "He": "nT", "Hn": "nT", "total": "nT"},
        instrument="goes_magnetometer",
        domain="satellite",
        metadata={
            "source": "NOAA SWPC / GOES primary satellite magnetometer",
            "url": url,
            "attribution": "Data: NOAA SWPC / GOES (public domain)",
            "fetched_at": _now_iso(),
        },
    )


# ------------------------------------------------------------------ Elexon GB grid

def parse_elexon_frequency(payload: dict) -> tuple[list, dict]:
    """Parse the Elexon Insights system-frequency JSON response."""
    records = payload.get("data", [])
    times, freq = [], []
    for rec in records:
        try:
            t = datetime.fromisoformat(rec["measurementTime"].replace("Z", ""))
            f = float(rec["frequency"])
        except (KeyError, TypeError, ValueError, AttributeError):
            continue
        times.append(t)
        freq.append(f)
    order = np.argsort(np.array(times))
    times = [times[i] for i in order]
    freq = [freq[i] for i in order]
    return times, {"frequency": freq}


def fetch_gb_grid_frequency(hours: float = 24.0, timeout: float = 30.0) -> SignalFrame:
    """Great Britain grid frequency at 15-second resolution from the
    Elexon Insights API (nominal 50 Hz)."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=hours)
    url = ("https://data.elexon.co.uk/bmrs/api/v1/system/frequency"
           f"?from={start.strftime('%Y-%m-%dT%H:%M:%SZ')}"
           f"&to={end.strftime('%Y-%m-%dT%H:%M:%SZ')}&format=json")
    payload = json.loads(_http_get(url, timeout))
    times, data = parse_elexon_frequency(payload)
    return _finish(
        time=pd.DatetimeIndex(times),
        values=data,
        name="GB grid frequency (live)",
        units={"frequency": "Hz"},
        instrument="grid_frequency_sensor",
        domain="energy",
        metadata={
            "source": "Elexon Insights (GB electricity system operator data)",
            "url": "https://data.elexon.co.uk",
            "attribution": "Contains BMRS data © Elexon Limited, 2026 (free licence)",
            "nominal_frequency": 50.0,
            "fetched_at": _now_iso(),
        },
    )


# ------------------------------------------------------------------ registry

LIVE_SOURCES = {
    "plasma: solar wind at L1 (live)": fetch_solar_wind,
    "ocean: NDBC buoy 46042 (live)": fetch_ndbc_buoy,
    "satellite: GOES magnetometer (live)": fetch_goes_magnetometer,
    "energy: GB grid frequency (live)": fetch_gb_grid_frequency,
}


def fetch_all_live(timeout: float = 30.0,
                   strict: bool = False) -> dict[str, SignalFrame]:
    """Fetch every live source. With ``strict=False`` (default), sources
    that fail are skipped; with ``strict=True`` the first failure raises."""
    out: dict[str, SignalFrame] = {}
    for name, fetcher in LIVE_SOURCES.items():
        try:
            out[name] = fetcher(timeout=timeout)
        except SourceUnavailable:
            if strict:
                raise
    return out
