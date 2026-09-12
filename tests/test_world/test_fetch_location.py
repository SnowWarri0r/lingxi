"""The fetch has to know which city is outside her window.

The weather block, the daylight calc and the fetch all describe "外面", and
only the first two read persona.location. Left to the topic list alone, the
first day's real output held both 「外面一直在下雨 24度」 (fetched as local to
Japan) and 「上海那边台风刚走完又来一个」 (Shanghai as elsewhere), while the
weather line in the same prompt was Shanghai's — one city as both 外面 and
那边.
"""

from datetime import date

from lingxi.persona.models import Identity, LocationConfig, PersonaConfig
from lingxi.world.fetcher import build_fetch_prompt


def _persona(location=None):
    return PersonaConfig(
        name="唐可可", id="tangkeke",
        identity=Identity(full_name="唐可可", age=21),
        world_interests=["音乐 / 现场演出"],
        location=location)


def test_the_prompt_names_the_city_she_is_in():
    p = _persona(LocationConfig(name="上海", latitude=31.23, longitude=121.47,
                                utc_offset=8))
    assert "上海" in build_fetch_prompt(p, date(2026, 9, 8))


def test_a_different_city_follows_the_config():
    p = _persona(LocationConfig(name="东京", latitude=35.68, longitude=139.65,
                                utc_offset=9))
    prompt = build_fetch_prompt(p, date(2026, 9, 8))

    assert "东京" in prompt
    assert "上海" not in prompt


def test_a_persona_without_a_location_still_builds_a_prompt():
    assert build_fetch_prompt(_persona(), date(2026, 9, 8)) is not None
