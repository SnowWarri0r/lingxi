"""A restart does not write a moment of its own.

The executor ticked as soon as the bot started: two deploys on one evening
put three moments into 27 minutes.
"""

from datetime import datetime, timedelta

import pytest

from lingxi.facts.models import Fact, FactType, Source
from lingxi.planner.executor import PlanExecutor

NOW = datetime(2026, 3, 9, 20, 40)


class _Retriever:
    def __init__(self, *minutes_ago):
        self.facts = [Fact(subject="aria", content="写词", source=Source.LIFE_SIMULATED,
                           type=FactType.EVENT, ts=NOW - timedelta(minutes=m))
                      for m in minutes_ago]

    async def fetch(self, q):
        return [f for f in self.facts if q.since is None or f.ts >= q.since]


def _executor(retriever):
    return PlanExecutor(llm=None, retriever=retriever, life_writer=None)


@pytest.mark.asyncio
async def test_after_a_restart_the_tick_waits_out_the_interval():
    assert await _executor(_Retriever(19, 49)).first_tick_in(NOW) == 11 * 60


@pytest.mark.asyncio
async def test_with_no_recent_moment_it_ticks_at_once():
    assert await _executor(_Retriever(45)).first_tick_in(NOW) == 0
    assert await _executor(_Retriever()).first_tick_in(NOW) == 0
