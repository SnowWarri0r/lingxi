"""What she has to talk about should be her day, not her last three hours.

The life sim emits a moment every half hour, and the block took the six most
recent — so it was always one three-hour stretch. Measured on 2026-08-30 it
spanned 17:28–19:59: one evening run by the river, her entire stock. It shows
in the openers: sampling that path, 11 of 20 led with the single rehearsal in
her recent events, phrased eleven different ways.
"""

from datetime import datetime, timedelta

from lingxi.facts.models import Fact, FactType, Source
from lingxi.temporal.proactive import select_spread_events


NOW = datetime(2026, 8, 30, 20, 0)


def _events(*minutes_ago: float) -> list[Fact]:
    return [
        Fact(subject="aria", content=f"第 {m} 分钟前那件事",
             source=Source.LIFE_SIMULATED, type=FactType.EVENT,
             ts=NOW - timedelta(minutes=m))
        for m in minutes_ago
    ]


def _span_hours(picked: list[Fact]) -> float:
    return (picked[0].ts - picked[-1].ts).total_seconds() / 3600


def test_a_half_hourly_day_is_spread_not_truncated():
    """Twenty ticks, thirty minutes apart — ten hours of material."""
    day = _events(*[i * 30 for i in range(20)])

    picked = select_spread_events(day, k=6)

    assert len(picked) == 6
    assert _span_hours(picked) >= 7.5, "the picks cover the whole day"


def test_the_newest_moment_is_always_kept():
    """「你此刻在干嘛」 has to stay right."""
    day = _events(*[i * 30 for i in range(20)])

    assert select_spread_events(day, k=6)[0].ts == NOW


def test_picks_are_returned_newest_first():
    picked = select_spread_events(_events(*[i * 30 for i in range(20)]), k=6)
    assert picked == sorted(picked, key=lambda f: f.ts, reverse=True)


def test_the_picks_are_evenly_spread_not_clustered():
    """Six picks over a 9.5-hour day land roughly 1.5h apart, none adjacent."""
    day = _events(*[i * 30 for i in range(20)])

    picked = select_spread_events(day, k=6)

    gaps = [(picked[i].ts - picked[i + 1].ts).total_seconds() / 3600
            for i in range(len(picked) - 1)]
    assert min(gaps) >= 1.0, gaps
    assert max(gaps) - min(gaps) <= 1.0, f"uneven: {gaps}"


def test_a_quiet_day_still_fills_the_block():
    """Four events an hour apart must not come back as two.

    Handing back a near-empty block would be worse than the concentration it
    replaces — she would have nothing at all to open with.
    """
    picked = select_spread_events(_events(0, 60, 120, 180), k=6)

    assert len(picked) == 4


def test_a_single_event_survives():
    assert len(select_spread_events(_events(0), k=6)) == 1


def test_no_events_is_not_an_error():
    assert select_spread_events([], k=6) == []


def test_it_reaches_back_into_yesterday_when_today_is_thin():
    """Two moments today, the rest last night — she can still say something."""
    picked = select_spread_events(_events(0, 30, 900, 960, 1020, 1080), k=6)

    assert len(picked) >= 4
    assert picked[0].ts == NOW


def test_the_old_behaviour_would_have_failed_this():
    """Guards the actual regression: the six newest span three hours."""
    day = _events(*[i * 30 for i in range(20)])

    naive = sorted(day, key=lambda f: f.ts, reverse=True)[:6]

    assert _span_hours(naive) == 2.5
    assert _span_hours(select_spread_events(day, k=6)) > _span_hours(naive)
