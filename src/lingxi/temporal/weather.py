"""Current weather for a persona's location (Open-Meteo, keyless & free).

Open-Meteo needs no API key and is reachable domestically without a proxy,
and it takes the same lat/lon the persona already carries for sun times.

The prompt is assembled synchronously, but a weather fetch is async network
I/O — so this module keeps a small in-process cache that a background loop
refreshes on an interval. The prompt path reads the cached value with a
plain sync call and never blocks; a failed or missing fetch simply yields
no weather line (the chat is never held up or broken by weather).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import httpx

from lingxi.temporal.sun import Location


_ENDPOINT = "https://api.open-meteo.com/v1/forecast"
_TTL = timedelta(minutes=30)
_TIMEOUT = 8.0

# WMO weather interpretation codes → short Chinese description.
# https://open-meteo.com/en/docs (weather_code table)
_WMO_ZH: dict[int, str] = {
    0: "晴",
    1: "大致晴朗", 2: "多云", 3: "阴",
    45: "有雾", 48: "雾凇",
    51: "毛毛雨", 53: "小雨", 55: "中雨",
    56: "冻毛毛雨", 57: "冻雨",
    61: "小雨", 63: "中雨", 65: "大雨",
    66: "冻雨", 67: "大冻雨",
    71: "小雪", 73: "中雪", 75: "大雪", 77: "雪粒",
    80: "阵雨", 81: "强阵雨", 82: "暴雨",
    85: "阵雪", 86: "强阵雪",
    95: "雷阵雨", 96: "雷阵雨伴冰雹", 99: "强雷暴伴冰雹",
}


@dataclass(frozen=True)
class Weather:
    temp_c: float
    feels_like_c: float
    description: str          # Chinese, from WMO code
    wind_kmh: float
    is_day: bool
    fetched_at: datetime

    def phrase(self) -> str:
        """One-line, plain-fact weather for the prompt."""
        parts = [f"{self.description}", f"{round(self.temp_c)}°C"]
        # Surface feels-like only when it diverges enough to matter.
        if abs(self.feels_like_c - self.temp_c) >= 3:
            parts.append(f"体感 {round(self.feels_like_c)}°C")
        if self.wind_kmh >= 25:
            parts.append("风挺大")
        return "，".join(parts)


@dataclass(frozen=True)
class DayOutlook:
    """Today as a whole, for deciding a day rather than describing a moment.

    A plan is written at 07:00, when the current reading is the day's low and
    says nothing about the afternoon it is scheduling. Rain is the part that
    actually decides indoor or outdoor, and it is the part a current-conditions
    reading cannot see at all.
    """

    high_c: float
    low_c: float
    precip_mm: float
    description: str
    fetched_at: datetime

    def phrase(self) -> str:
        parts = [f"{self.description}",
                 f"最高 {round(self.high_c)}°C",
                 f"最低 {round(self.low_c)}°C"]
        if self.precip_mm >= 5:
            parts.append(f"一天下来有 {self.precip_mm:.0f}mm 的雨")
        elif self.precip_mm >= 0.5:
            parts.append("零星有点雨")
        else:
            parts.append("没雨")
        return "，".join(parts)


# Descriptions under which the sun is out, or out often enough to say so.
# 多云 is partly cloudy — the sun comes and goes, so a sunny line still holds.
_SUN_OUT = frozenset({"晴", "大致晴朗", "多云"})


def sun_visible(weather: "Weather | None") -> bool | None:
    """Whether the sky lets the sun through, or None when nothing is known.

    Daylight is computed from sunrise and sunset alone, which says when the
    sun is up, not whether it can be seen. On 2026-09-24 the prompt carried
    「日头正好——晒太阳、趴窗台这类成立」 two lines above 「外面天气：阴」,
    and the opener she wrote was about the sun failing to show up.
    """
    if weather is None or weather.description not in _WMO_ZH.values():
        return None
    return weather.description in _SUN_OUT


# Cache keyed by rounded (lat, lon) so nearby coords share an entry.
_cache: dict[tuple[float, float], Weather] = {}
_outlook_cache: dict[tuple[float, float], DayOutlook] = {}


def _key(loc: Location) -> tuple[float, float]:
    return (round(loc.latitude, 2), round(loc.longitude, 2))


def cached(loc: Location, *, now: datetime | None = None) -> Weather | None:
    """Fresh cached weather for the location, or None. Sync, non-blocking."""
    w = _cache.get(_key(loc))
    if w is None:
        return None
    now = now or datetime.now()
    if now - w.fetched_at > _TTL:
        return None
    return w


def cached_outlook(loc: Location, *, now: datetime | None = None) -> DayOutlook | None:
    """Today's outlook, if one was fetched today. Sync, non-blocking."""
    o = _outlook_cache.get(_key(loc))
    if o is None:
        return None
    now = now or datetime.now()
    if o.fetched_at.date() != now.date():
        return None
    return o


def _parse_outlook(payload: dict, now: datetime) -> DayOutlook | None:
    daily = payload.get("daily")
    if not isinstance(daily, dict):
        return None
    try:
        return DayOutlook(
            high_c=float(daily["temperature_2m_max"][0]),
            low_c=float(daily["temperature_2m_min"][0]),
            precip_mm=float(daily.get("precipitation_sum", [0.0])[0] or 0.0),
            description=_WMO_ZH.get(int(daily["weather_code"][0]), "天气未知"),
            fetched_at=now,
        )
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def _parse(payload: dict, now: datetime) -> Weather | None:
    cur = payload.get("current")
    if not isinstance(cur, dict) or "temperature_2m" not in cur:
        return None
    code = int(cur.get("weather_code", -1))
    return Weather(
        temp_c=float(cur["temperature_2m"]),
        feels_like_c=float(cur.get("apparent_temperature", cur["temperature_2m"])),
        description=_WMO_ZH.get(code, "天气未知"),
        wind_kmh=float(cur.get("wind_speed_10m", 0.0)),
        is_day=bool(cur.get("is_day", 1)),
        fetched_at=now,
    )


async def refresh(loc: Location, *, now: datetime | None = None) -> Weather | None:
    """Fetch current weather and update the cache. Fail-safe: on any error
    returns None and leaves any existing cache entry untouched."""
    now = now or datetime.now()
    params = {
        "latitude": loc.latitude,
        "longitude": loc.longitude,
        "current": ("temperature_2m,apparent_temperature,weather_code,"
                    "wind_speed_10m,is_day"),
        # Same request, same cost — the day's own figures ride along with the
        # moment's, so anything planning a whole day has them to read.
        "daily": ("temperature_2m_max,temperature_2m_min,"
                  "precipitation_sum,weather_code"),
        "forecast_days": 1,
        "timezone": "auto",
    }
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(_ENDPOINT, params=params)
            resp.raise_for_status()
            payload = resp.json()
            weather = _parse(payload, now)
            outlook = _parse_outlook(payload, now)
    except Exception as e:
        print(f"[weather] refresh failed for {loc.name or _key(loc)}: {e}",
              flush=True)
        return None
    if weather is not None:
        _cache[_key(loc)] = weather
    if outlook is not None:
        _outlook_cache[_key(loc)] = outlook
    return weather
