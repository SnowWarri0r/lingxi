"""Her reflections no longer steer the day.

Classified by the main model, 57-84% of her reflections since early August
taught one lesson — 别想太多、信身体和当下、直接去做、留白、放手. Fed to the
planner, they made every plan an enactment of it: on the real 09-24 prompt,
37 of 37 plan items with them, 11 of 26 with none. Reflections are still
written, and still reachable from chat as aria.pattern.
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

LESSON = "身体比脑子先知道——往后不用老问稳了没，信号来了再落"


class _LLM:
    def __init__(self, items=None):
        self.prompts = []
        self._items = items or [{"time_window": "09:00-10:00", "content": "去练习室跑第二段"}]

    async def complete(self, *, messages, system=None, **kw):
        self.prompts.append((system or "") + messages[0]["content"])
        return SimpleNamespace(content=json.dumps(self._items, ensure_ascii=False))


async def _planner(tmp_path, llm, patterns=()):
    store = FactStore(tmp_path / "facts.db")
    await store.init()
    for content, days_ago in patterns:
        await store.write(Fact(subject="aria", content=content, source=Source.LLM_INFERRED,
                               type=FactType.PATTERN, importance=8,
                               ts=datetime.now() - timedelta(days=days_ago)))
    p = DailyPlanner(llm, FactRetriever(store), LifeWriter(store, scorer=None))

    async def _no_weather(now):
        return ""
    p._todays_weather = _no_weather
    return p, store


@pytest.mark.asyncio
async def test_her_reflections_do_not_reach_the_plan_prompt(tmp_path):
    llm = _LLM()
    planner, _ = await _planner(tmp_path, llm, [(LESSON, 0), (LESSON + "（又一次）", 3)])

    await planner.plan_aria()

    assert LESSON not in llm.prompts[0]
    assert "反思" not in llm.prompts[0] and "模式" not in llm.prompts[0]


@pytest.mark.asyncio
async def test_a_day_is_still_planned_and_written(tmp_path):
    llm = _LLM()
    planner, store = await _planner(tmp_path, llm)

    written = await planner.plan_aria()

    assert [f.content for f in written] == ["去练习室跑第二段"]
    assert await store.query(subject="aria", type=FactType.PLAN, limit=5)


@pytest.mark.asyncio
async def test_the_people_in_her_life_still_reach_it(tmp_path):
    llm = _LLM()
    planner, _ = await _planner(tmp_path, llm)

    await planner.plan_aria()

    assert "【我生活里的人】" in llm.prompts[0]
