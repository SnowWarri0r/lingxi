"""A day whose plan failed at 07:00 is planned again later, not left empty."""

from datetime import datetime, timedelta

import pytest

from lingxi.planner.daily_planner import DailyPlanner


class _Clock:
    def __init__(self, t):
        self.t = t

    def __call__(self):
        return self.t


def _planner(results):
    p = DailyPlanner.__new__(DailyPlanner)
    calls = []

    async def plan_aria():
        calls.append(1)
        return results.pop(0)

    p.plan_aria = plan_aria
    return p, calls


@pytest.mark.asyncio
async def test_a_refused_morning_is_tried_again_until_a_plan_is_written():
    p, calls = _planner([[], [], ["plan"]])
    clock = _Clock(datetime(2026, 11, 22, 7, 0))
    slept = []

    async def sleep(s):
        slept.append(s)
        clock.t += timedelta(seconds=s)

    assert await p.plan_aria_until_written(clock=clock, sleep=sleep) == ["plan"]
    assert len(calls) == 3 and slept == [1800, 1800]


@pytest.mark.asyncio
async def test_a_plan_written_first_time_is_not_asked_for_again():
    p, calls = _planner([["plan"]])

    async def sleep(s):
        raise AssertionError("slept")

    await p.plan_aria_until_written(clock=_Clock(datetime(2026, 11, 22, 7)), sleep=sleep)
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_late_in_the_day_it_stops_trying():
    p, calls = _planner([[]] * 10)
    clock = _Clock(datetime(2026, 11, 22, 19, 0))

    async def sleep(s):
        clock.t += timedelta(seconds=s)

    assert await p.plan_aria_until_written(clock=clock, sleep=sleep) == []
    # 19:00, 19:30, 20:00 tried; 20:30 + 30 min reaches 21:00.
    assert len(calls) == 4
