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


# Cache keyed by rounded (lat, lon) so nearby coords share an entry.
_cache: dict[tuple[float, float], Weather] = {}


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
        "timezone": "auto",
    }
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(_ENDPOINT, params=params)
            resp.raise_for_status()
            weather = _parse(resp.json(), now)
    except Exception as e:
        print(f"[weather] refresh failed for {loc.name or _key(loc)}: {e}",
              flush=True)
        return None
    if weather is not None:
        _cache[_key(loc)] = weather
    return weather
