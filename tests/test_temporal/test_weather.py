from datetime import datetime, timedelta

import pytest

from lingxi.temporal.sun import Location
from lingxi.temporal import weather
from lingxi.temporal.weather import Weather


LOC = Location("上海", 31.2304, 121.4737, 8.0)


def _mk(temp, feels, code, wind, at):
    return Weather(temp_c=temp, feels_like_c=feels,
                   description=weather._WMO_ZH.get(code, "天气未知"),
                   wind_kmh=wind, is_day=True, fetched_at=at)


def test_parse_maps_wmo_code_to_chinese():
    now = datetime(2026, 7, 21, 14, 0)
    payload = {"current": {"temperature_2m": 30.1, "apparent_temperature": 34.0,
                           "weather_code": 61, "wind_speed_10m": 5.0, "is_day": 1}}
    w = weather._parse(payload, now)
    assert w.description == "小雨"
    assert round(w.temp_c) == 30


def test_parse_returns_none_on_malformed():
    assert weather._parse({}, datetime.now()) is None
    assert weather._parse({"current": {"weather_code": 0}}, datetime.now()) is None


def test_phrase_shows_feels_like_only_when_diverging():
    at = datetime.now()
    near = _mk(28.0, 29.0, 0, 5.0, at)      # <3° gap → no 体感
    far = _mk(28.0, 34.0, 0, 5.0, at)       # ≥3° gap → 体感 shown
    assert "体感" not in near.phrase()
    assert "体感 34°C" in far.phrase()


def test_phrase_flags_strong_wind():
    at = datetime.now()
    assert "风挺大" in _mk(15.0, 15.0, 3, 30.0, at).phrase()
    assert "风挺大" not in _mk(15.0, 15.0, 3, 10.0, at).phrase()


def test_cache_respects_ttl():
    weather._cache.clear()
    now = datetime(2026, 7, 21, 14, 0)
    weather._cache[weather._key(LOC)] = _mk(27.0, 27.0, 0, 3.0, now)
    # within TTL → returned; past TTL → None
    assert weather.cached(LOC, now=now + timedelta(minutes=20)) is not None
    assert weather.cached(LOC, now=now + timedelta(minutes=45)) is None
    weather._cache.clear()
