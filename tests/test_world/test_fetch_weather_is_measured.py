"""The scan searched for the weather and got it wrong on the days it mattered.

「她所在城市的身边事」 reliably produces a weather item, and it was written
from web search. Checked against the Shanghai record: 09-19 was scanned as
「放晴了…出门不用带伞」 on the wettest day of the fortnight (9.5mm of rain),
and 09-08 as 「外面一直在下雨」 on a day with none. That sentence is then read
in the chat prompt directly beside the real Open-Meteo reading, which the
system has fetched every twenty minutes the whole time.

So the scan is handed the figures instead of sent looking for them.
"""

from datetime import date

from lingxi.persona.models import Identity, LocationConfig, PersonaConfig
from lingxi.world.fetcher import build_fetch_prompt


SHANGHAI = LocationConfig(name="上海", latitude=31.23, longitude=121.47,
                          utc_offset=8)
WET = "中雨，最高 31°C，最低 24°C，一天下来有 10mm 的雨"


def _persona():
    return PersonaConfig(
        name="唐可可", id="tangkeke",
        identity=Identity(full_name="唐可可", age=21),
        world_interests=["她所在城市的身边事"],
        location=SHANGHAI)


def _prompt(weather=""):
    return build_fetch_prompt(_persona(), date(2026, 9, 19), weather=weather)


def test_the_measured_weather_is_in_the_prompt():
    assert WET in _prompt(WET)


def test_it_is_marked_as_measured():
    assert "实测" in _prompt(WET)


def test_the_scan_is_told_to_write_from_those_figures():
    after = _prompt(WET).split(WET, 1)[1][:60]

    assert "照这个" in after or "按这个" in after


def test_no_reading_leaves_the_prompt_as_it_was():
    """Network down, no location, malformed response — the scan still runs."""
    bare = _prompt()

    assert "实测" not in bare
    assert "她所在城市的身边事" in bare


def test_a_blank_reading_is_not_a_reading():
    assert "实测" not in _prompt("   ")


def test_the_reading_sits_above_the_selection_criteria():
    """It constrains what gets written, so it is read before the picking."""
    prompt = _prompt(WET)

    assert prompt.index(WET) < prompt.index("挑选标准")
