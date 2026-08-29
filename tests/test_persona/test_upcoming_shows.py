"""Booked dates carry a computed distance, not a bare month."""

from datetime import date

from lingxi.persona.models import UpcomingShow
from lingxi.persona.prompt_builder import _gap_cn, _upcoming_shows_block

TODAY = date(2026, 8, 24)


def test_a_november_show_is_not_tomorrow():
    """The regression this exists for.

    The schedule was prose reading 「11 月」 with nothing saying how far off
    that was, and on 8月24日 she announced 「明天就是我名古屋 Love Live Fes
    的正式舞台了」.
    """
    block = _upcoming_shows_block(
        [UpcomingShow(event="Love Live Fes 15th", date="2026-11",
                      venue="バンテリンドーム ナゴヤ")], TODAY)
    assert "还有 2 个多月" in block
    assert "就是明天" not in block
    assert "2026年11月" in block


def test_month_only_dates_do_not_invent_a_day():
    """Only the month is known, so only the month is stated."""
    block = _upcoming_shows_block([UpcomingShow(event="X", date="2026-11")], TODAY)
    entry = next(ln for ln in block.splitlines() if ln.startswith("- "))
    assert "2026年11月——" in entry, entry


def test_precise_dates_render_the_day():
    block = _upcoming_shows_block(
        [UpcomingShow(event="X", date="2026-09-10")], TODAY)
    assert "2026年9月10日" in block
    assert "还有 17 天" in block


def test_gap_wording_across_the_range():
    assert _gap_cn(date(2026, 8, 24), TODAY) == "**就是今天**"
    assert _gap_cn(date(2026, 8, 25), TODAY) == "**就是明天**"
    assert _gap_cn(date(2026, 9, 3), TODAY) == "还有 10 天"
    assert "个多月" in _gap_cn(date(2026, 11, 21), TODAY)
    assert "已经过去" in _gap_cn(date(2026, 7, 1), TODAY)


def test_no_shows_renders_nothing():
    assert _upcoming_shows_block([], TODAY) is None
    assert _upcoming_shows_block(None, TODAY) is None


def test_an_unparseable_date_is_skipped_not_crashed():
    """A typo must not take the whole prompt down."""
    block = _upcoming_shows_block(
        [UpcomingShow(event="坏的", date="十一月"),
         UpcomingShow(event="好的", date="2026-11")], TODAY)
    assert "好的" in block
    assert "坏的" not in block


def test_the_live_persona_states_the_distance():
    from lingxi.persona.loader import load_persona
    from lingxi.persona.prompt_builder import build_persona_block

    b = build_persona_block(load_persona("config/personas/tangkeke.yaml"))
    assert "接下来你要演的场" in b
    assert "别自己估" in b
