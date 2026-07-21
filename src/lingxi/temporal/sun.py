"""Offline sunrise/sunset computation (NOAA solar algorithm).

Pure math, no network and no dependency — given a location and a date it
returns local sunrise/sunset wall-clock times, accurate to ~1 minute for
non-polar latitudes. Used to ground a persona's sense of daylight in the
real season and region instead of hardcoded hour buckets (winter sunset at
17:00 vs summer at 19:00 is a real difference the persona should feel).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta


@dataclass(frozen=True)
class Location:
    name: str
    latitude: float
    longitude: float          # east positive
    utc_offset_hours: float = 8.0   # persona's local zone; China default


_DEFAULT_LOCATION = Location("上海", 31.2304, 121.4737, 8.0)


def persona_location(persona) -> Location:
    """Resolve a persona's Location, or the domestic default when its YAML
    leaves `location` unset."""
    loc = getattr(persona, "location", None) if persona is not None else None
    if loc is None:
        return _DEFAULT_LOCATION
    return Location(
        name=loc.name, latitude=loc.latitude,
        longitude=loc.longitude, utc_offset_hours=loc.utc_offset,
    )


@dataclass(frozen=True)
class SunTimes:
    sunrise: datetime | None   # local wall clock; None on polar night/day
    sunset: datetime | None
    polar_day: bool = False    # sun never sets
    polar_night: bool = False  # sun never rises


_ZENITH_DEG = 90.833   # standard refraction + solar disc radius


def sun_times(loc: Location, on: date) -> SunTimes:
    """Local sunrise/sunset for the location on the given date."""
    n = on.timetuple().tm_yday
    # Fractional year (radians), evaluated at solar noon.
    gamma = 2.0 * math.pi / 365.0 * (n - 1 + 0.5)

    eqtime = 229.18 * (
        0.000075
        + 0.001868 * math.cos(gamma)
        - 0.032077 * math.sin(gamma)
        - 0.014615 * math.cos(2 * gamma)
        - 0.040849 * math.sin(2 * gamma)
    )
    decl = (
        0.006918
        - 0.399912 * math.cos(gamma)
        + 0.070257 * math.sin(gamma)
        - 0.006758 * math.cos(2 * gamma)
        + 0.000907 * math.sin(2 * gamma)
        - 0.002697 * math.cos(3 * gamma)
        + 0.00148 * math.sin(3 * gamma)
    )

    lat_rad = math.radians(loc.latitude)
    cos_ha = (
        math.cos(math.radians(_ZENITH_DEG))
        / (math.cos(lat_rad) * math.cos(decl))
        - math.tan(lat_rad) * math.tan(decl)
    )
    if cos_ha < -1.0:
        return SunTimes(sunrise=None, sunset=None, polar_day=True)
    if cos_ha > 1.0:
        return SunTimes(sunrise=None, sunset=None, polar_night=True)

    ha_deg = math.degrees(math.acos(cos_ha))
    # UTC minutes from midnight (NOAA); lon east positive.
    sunrise_utc_min = 720.0 - 4.0 * (loc.longitude + ha_deg) - eqtime
    sunset_utc_min = 720.0 - 4.0 * (loc.longitude - ha_deg) - eqtime
    offset_min = loc.utc_offset_hours * 60.0

    return SunTimes(
        sunrise=_local_dt(on, sunrise_utc_min + offset_min),
        sunset=_local_dt(on, sunset_utc_min + offset_min),
    )


def _local_dt(on: date, minutes_from_midnight: float) -> datetime:
    """Clamp into the day and build a local datetime (minute precision)."""
    minutes = max(0.0, min(24 * 60 - 1, minutes_from_midnight))
    h = int(minutes // 60)
    m = int(round(minutes - h * 60))
    if m == 60:
        h, m = h + 1, 0
    h = min(h, 23)
    return datetime.combine(on, time(hour=h, minute=m))
