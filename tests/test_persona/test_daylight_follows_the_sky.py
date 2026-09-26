"""Sun times say when the sun is up, not whether anyone can see it.

On 2026-09-24 the opener prompt carried, two lines apart,
「大白天，日头正好——晒太阳、趴窗台这类成立」 and 「外面天气：阴，26°C」.
The opener she wrote was about the sun not showing up — resolving the
contradiction out loud — and it was the third cloud opener in a row.
"""

from datetime import datetime

import pytest

from lingxi.persona.models import Identity, LocationConfig, PersonaConfig
from lingxi.persona.prompt_builder import PromptBuilder
from lingxi.temporal import weather as wx

SHANGHAI = LocationConfig(name="上海", latitude=31.23, longitude=121.47, utc_offset=8)
NOON = datetime(2026, 9, 24, 12, 0)
BEFORE_SUNSET = datetime(2026, 9, 24, 17, 20)


def _builder():
    return PromptBuilder(PersonaConfig(name="唐可可", id="tangkeke",
                                       identity=Identity(full_name="唐可可"), location=SHANGHAI))


@pytest.fixture
def sky():
    key = (31.23, 121.47)
    saved = wx._cache.get(key)

    def _set(desc, at):
        wx._cache[key] = wx.Weather(temp_c=26, feels_like_c=26, description=desc,
                                    wind_kmh=5, is_day=True, fetched_at=at)
    yield _set
    if saved is None:
        wx._cache.pop(key, None)
    else:
        wx._cache[key] = saved


def test_an_overcast_noon_does_not_promise_sunshine(sky):
    sky("阴", NOON)
    scene = _builder()._daylight_scene(NOON)

    assert "日头正好" not in scene and "晒太阳" not in scene
    assert "阴" in scene


def test_rain_is_the_same(sky):
    sky("小雨", NOON)
    assert "晒太阳" not in _builder()._daylight_scene(NOON)


def test_an_overcast_evening_has_no_sunset_to_watch(sky):
    sky("阴", BEFORE_SUNSET)
    scene = _builder()._daylight_scene(BEFORE_SUNSET)

    assert "晚霞" not in scene and "夕阳" not in scene


def test_a_clear_noon_still_says_the_sun_is_out(sky):
    sky("晴", NOON)
    assert "日头正好" in _builder()._daylight_scene(NOON)


def test_partly_cloudy_keeps_the_sun(sky):
    """多云: the sun comes and goes, so the sunny line still holds."""
    sky("多云", NOON)
    assert "日头正好" in _builder()._daylight_scene(NOON)


def test_no_reading_leaves_the_line_as_it_was(sky):
    """Weather feed down: say what the sun times say, as before."""
    wx._cache.pop((31.23, 121.47), None)
    assert "日头正好" in _builder()._daylight_scene(NOON)


def test_an_unrecognised_sky_claims_nothing_either_way(sky):
    sky("天气未知", NOON)
    scene = _builder()._daylight_scene(NOON)

    assert "天气未知" not in scene
