"""Her day was planned entirely from the inside.

Everything the planner reads is her own prior output or static YAML —
yesterday's reflections, the week's patterns, the people in the persona file.
Nothing she had not already thought of could reach the day.

Meanwhile a world briefing is fetched every morning at 06:03 and the plan is
written at 07:00 — that order held on 55 of 63 days — and the two were never
joined. Measured consequence: the 09-11 scan said the typhoon's outer band
would be worst on the 13th and 14th; the 13th was planned around an aimless
walk and a sunny window seat, and the 14th around eating breakfast on the
steps outside.
"""

import json
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from lingxi.facts.models import Fact, FactType, Source
from lingxi.facts.retriever import FactRetriever
from lingxi.facts.store import FactStore
from lingxi.facts.writers.life import LifeWriter
from lingxi.planner.daily_planner import DailyPlanner, build_outside_block


class FakeLLM:
    def __init__(self):
        self.prompts = []

    async def complete(self, *, messages, system=None, **kw):
        self.prompts.append(messages[0]["content"])
        return SimpleNamespace(
            content=json.dumps([{"time_window": "09:00-10:00", "content": "x"}]))


def _world(content, hours_ago=1.0):
    return Fact(subject="world", content=content, source=Source.LLM_INFERRED,
                type=FactType.EVENT, ts=datetime.now() - timedelta(hours=hours_ago),
                importance=5)


async def _prompt_over(facts, tmp_path, weather=""):
    store = FactStore(tmp_path / "facts.db")
    await store.init()
    for f in facts:
        await store.write(f)
    llm = FakeLLM()
    planner = DailyPlanner(llm, FactRetriever(store), LifeWriter(store, scorer=None))
    # Stubbed: the real one reaches Open-Meteo, and a test suite that needs a
    # network is a test suite that goes red on a train.
    planner._todays_weather = lambda now: _returning(weather)
    await planner.plan_aria()
    return llm.prompts[0]


async def _returning(value):
    return value


TYPHOON = "今天就开始飘雨了 13号14号才是最猛的那两天 台风外围绕过来的"


@pytest.mark.asyncio
async def test_the_weather_reaches_the_day(tmp_path):
    assert TYPHOON in await _prompt_over([_world(TYPHOON)], tmp_path)


@pytest.mark.asyncio
async def test_the_day_is_told_to_be_planned_in_those_conditions(tmp_path):
    """Facts with no instruction attached read as trivia and get ignored."""
    prompt = await _prompt_over([_world(TYPHOON)], tmp_path)
    after = prompt.split(TYPHOON, 1)[1].split("【怎么安排】", 1)[0]

    assert "室内" in after and "外面" in after


@pytest.mark.asyncio
async def test_every_item_of_the_scan_is_carried(tmp_path):
    scan = [_world("今天33度 出门要防晒"),
            _world("达达今晚就在本地演"),
            _world("スクミュ要开单独live了 2027年1月")]
    prompt = await _prompt_over(scan, tmp_path)

    for f in scan:
        assert f.content in prompt


@pytest.mark.asyncio
async def test_yesterdays_scan_does_not_plan_today(tmp_path):
    """The briefing expires with its day; a stale one would forecast wrongly."""
    stale = _world(TYPHOON, hours_ago=30)
    assert TYPHOON not in await _prompt_over([stale], tmp_path)


@pytest.mark.asyncio
async def test_a_missing_scan_still_yields_a_plan(tmp_path):
    """8 of 63 days had no briefing in hand by 07:00."""
    prompt = await _prompt_over([], tmp_path)

    assert "【怎么安排】" in prompt and "【今天外面】" not in prompt


@pytest.mark.asyncio
async def test_the_weather_arrives_even_when_the_scan_does_not(tmp_path):
    """The two sources fail independently; either one alone is worth having."""
    prompt = await _prompt_over([], tmp_path, weather="中雨，最高 31°C，一天下来有 10mm 的雨")

    assert "10mm 的雨" in prompt


@pytest.mark.asyncio
async def test_the_measured_weather_leads_the_scan_in_the_real_prompt(tmp_path):
    prompt = await _prompt_over([_world(TYPHOON)], tmp_path,
                                weather="晴，最高 32°C，最低 22°C，没雨")

    assert prompt.index("32°C") < prompt.index(TYPHOON)


class TestTheBlockItself:
    def test_nothing_scanned_means_no_heading(self):
        assert build_outside_block([]) == ""

    def test_blank_content_is_not_a_scan(self):
        assert build_outside_block([_world("   ")]) == ""

    def test_the_block_ends_clear_of_what_follows(self):
        """It is spliced in directly above 【怎么安排】."""
        assert build_outside_block([_world(TYPHOON)]).endswith("\n\n")

    def test_real_weather_alone_is_enough_to_open_the_block(self):
        assert "最高 30°C" in build_outside_block([], "阴，最高 30°C，最低 22°C，没雨")

    def test_the_measured_weather_is_read_before_the_scan(self):
        """The scan's own weather sentence is written by a model and was wrong
        on the days it mattered; the measured figures come first."""
        block = build_outside_block([_world(TYPHOON)], "阴，最高 30°C，没雨")

        assert block.index("30°C") < block.index(TYPHOON)

    def test_the_measured_weather_is_marked_as_measured(self):
        block = build_outside_block([_world(TYPHOON)], "阴，最高 30°C，没雨")

        assert "实测" in block


class TestTodaysOutlook:
    """A plan written at 07:00 is a plan for the afternoon.

    The current reading at that hour is the day's low, and it cannot see rain
    coming at all — which is the half that decides indoor or outdoor.
    """

    def test_a_wet_day_says_so(self):
        from lingxi.temporal.weather import DayOutlook
        o = DayOutlook(high_c=31, low_c=24, precip_mm=9.5,
                       description="中雨", fetched_at=datetime.now())

        assert "9mm" in o.phrase() or "10mm" in o.phrase()

    def test_a_dry_day_says_so(self):
        from lingxi.temporal.weather import DayOutlook
        o = DayOutlook(high_c=32, low_c=22, precip_mm=0.0,
                       description="晴", fetched_at=datetime.now())

        assert "没雨" in o.phrase() and "32" in o.phrase()

    def test_the_high_is_carried_not_just_the_current_reading(self):
        from lingxi.temporal.weather import DayOutlook
        o = DayOutlook(high_c=30, low_c=22, precip_mm=0.0,
                       description="多云", fetched_at=datetime.now())

        assert "最高 30°C" in o.phrase() and "最低 22°C" in o.phrase()

    def test_yesterdays_outlook_is_not_todays(self):
        from lingxi.temporal.sun import Location
        from lingxi.temporal.weather import DayOutlook, _outlook_cache, cached_outlook
        loc = Location(latitude=31.23, longitude=121.47, name="上海")
        now = datetime.now()
        _outlook_cache[(31.23, 121.47)] = DayOutlook(
            high_c=30, low_c=22, precip_mm=0.0, description="晴",
            fetched_at=now - timedelta(days=1))
        try:
            assert cached_outlook(loc, now=now) is None
        finally:
            _outlook_cache.pop((31.23, 121.47), None)

    def test_a_response_without_a_daily_block_is_not_an_outlook(self):
        from lingxi.temporal.weather import _parse_outlook
        assert _parse_outlook({"current": {"temperature_2m": 29}}, datetime.now()) is None
