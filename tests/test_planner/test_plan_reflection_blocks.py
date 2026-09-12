"""The planner's two reflection blocks must not carry the same lines.

The prompt shows 【昨天我反思到的】 and 【最近一周我注意到的模式】 as if they
were different horizons. Both fetch PATTERN facts from one pool ranked
0.5*recency + 0.3*importance, so yesterday's insights win the week block too:
measured on the live store, all 5 of the reflection lines reappeared in the
10 pattern lines — 15 lines carrying 10 insights, with one day's theme at
double weight in a prompt that decides the whole next day.
"""

import json
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from lingxi.facts.models import Fact, FactType, Source
from lingxi.facts.retriever import FactRetriever
from lingxi.facts.store import FactStore
from lingxi.facts.writers.life import LifeWriter
from lingxi.planner.daily_planner import DailyPlanner


class FakeLLM:
    def __init__(self):
        self.prompts = []

    async def complete(self, *, messages, system=None, **kw):
        self.prompts.append(messages[0]["content"])
        return SimpleNamespace(
            content=json.dumps([{"time_window": "09:00-10:00", "content": "x"}]))


async def _planner_over(patterns, tmp_path):
    store = FactStore(tmp_path / "facts.db")
    await store.init()
    for p in patterns:
        await store.write(p)
    llm = FakeLLM()
    planner = DailyPlanner(llm, FactRetriever(store), LifeWriter(store, scorer=None))
    await planner.plan_aria()
    return llm.prompts[0]


def _pattern(content, days_ago, importance=7):
    ts = datetime.now() - timedelta(days=days_ago)
    return Fact(subject="aria", content=content, source=Source.LLM_INFERRED,
                type=FactType.PATTERN, ts=ts, importance=importance)


def _blocks(prompt):
    """(yesterday block, week block) as line lists."""
    after_y = prompt.split("【昨天我反思到的】", 1)[1]
    y_text, rest = after_y.split("【最近一周我注意到的模式】", 1)
    w_text = rest.split("【我生活里的人】", 1)[0]
    return ([ln.strip() for ln in y_text.strip().splitlines() if ln.strip()],
            [ln.strip() for ln in w_text.strip().splitlines() if ln.strip()])


@pytest.mark.asyncio
async def test_yesterdays_insights_do_not_reappear_in_the_week_block(tmp_path):
    patterns = [_pattern(f"昨天第{i}条", 0.4) for i in range(5)]
    patterns += [_pattern(f"前几天第{i}条", 2 + i * 0.5) for i in range(12)]

    prompt = await _planner_over(patterns, tmp_path)
    yesterday, week = _blocks(prompt)

    assert yesterday, "the yesterday block should not be empty"
    assert not (set(yesterday) & set(week)), (
        f"repeated between the two blocks: {set(yesterday) & set(week)}")


@pytest.mark.asyncio
async def test_the_week_block_stays_full_after_the_exclusion(tmp_path):
    """Dropping duplicates must not shrink the block — it reaches further back."""
    patterns = [_pattern(f"昨天第{i}条", 0.4) for i in range(5)]
    patterns += [_pattern(f"前几天第{i}条", 2 + i * 0.3) for i in range(12)]

    prompt = await _planner_over(patterns, tmp_path)
    _, week = _blocks(prompt)

    assert len(week) == 10


@pytest.mark.asyncio
async def test_the_two_blocks_together_carry_every_line_distinct(tmp_path):
    patterns = [_pattern(f"昨天第{i}条", 0.4) for i in range(5)]
    patterns += [_pattern(f"前几天第{i}条", 2 + i * 0.3) for i in range(12)]

    prompt = await _planner_over(patterns, tmp_path)
    yesterday, week = _blocks(prompt)
    lines = yesterday + week

    assert len(lines) == len(set(lines)) == 15


@pytest.mark.asyncio
async def test_a_thin_pool_still_plans(tmp_path):
    """Two insights total: the week block ends up empty, and that is fine."""
    prompt = await _planner_over(
        [_pattern("昨天唯一一条", 0.4), _pattern("再唯一一条", 0.5)], tmp_path)
    yesterday, week = _blocks(prompt)

    assert len(yesterday) == 2
    assert week == ["（最近没新模式）"]


@pytest.mark.asyncio
async def test_no_patterns_at_all_still_plans(tmp_path):
    prompt = await _planner_over([], tmp_path)
    yesterday, week = _blocks(prompt)

    assert yesterday == ["（昨天没特别的反思）"]
    assert week == ["（最近没新模式）"]
