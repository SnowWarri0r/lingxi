from datetime import date, datetime

from lingxi.temporal.sun import Location, sun_times


SHANGHAI = Location("上海", 31.2304, 121.4737, 8.0)


def test_summer_sunset_is_evening():
    st = sun_times(SHANGHAI, date(2026, 7, 20))
    # Shanghai midsummer: sunrise ~05:07, sunset ~19:00 (±few min tolerance).
    assert 4 * 60 + 50 <= _mins(st.sunrise) <= 5 * 60 + 20
    assert 18 * 60 + 45 <= _mins(st.sunset) <= 19 * 60 + 10


def test_winter_sunset_is_much_earlier_than_summer():
    summer = sun_times(SHANGHAI, date(2026, 7, 20))
    winter = sun_times(SHANGHAI, date(2026, 12, 21))
    # The whole point: winter goes dark far earlier than summer.
    assert _mins(winter.sunset) < _mins(summer.sunset) - 90
    assert 16 * 60 + 40 <= _mins(winter.sunset) <= 17 * 60 + 15


def test_polar_night_and_day():
    # High north in midsummer → sun never sets; high south → never rises.
    north = sun_times(Location("北", 78.0, 15.0, 1.0), date(2026, 7, 20))
    south = sun_times(Location("南", -78.0, 15.0, 1.0), date(2026, 7, 20))
    assert north.polar_day and north.sunset is None
    assert south.polar_night and south.sunrise is None


def test_utc_offset_shifts_wall_clock():
    east = sun_times(Location("A", 31.23, 121.47, 8.0), date(2026, 7, 20))
    west = sun_times(Location("A", 31.23, 121.47, 0.0), date(2026, 7, 20))
    # Same place, 8h less offset → sunset wall-clock 8h earlier.
    assert abs((_mins(east.sunset) - _mins(west.sunset)) - 8 * 60) <= 2


def _mins(dt: datetime) -> int:
    return dt.hour * 60 + dt.minute
