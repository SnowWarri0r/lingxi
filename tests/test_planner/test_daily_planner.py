import json
import pytest
from datetime import datetime, timedelta
from types import SimpleNamespace
from lingxi.facts.models import Fact, Source, FactType


class FakeLLM:
    def __init__(self, response: str):
        self.response = response
        self.calls = []
        self.systems = []

    async def complete(self, *, messages, system=None, **kw):
        self.calls.append(messages[0]["content"])
        self.systems.append(system or "")
        return SimpleNamespace(content=self.response)


@pytest.mark.asyncio
async def test_plan_aria_writes_plan_facts(tmp_path):
    from lingxi.facts.store import FactStore
    from lingxi.facts.retriever import FactRetriever
    from lingxi.facts.writers.life import LifeWriter
    from lingxi.planner.daily_planner import DailyPlanner

    plan_json = json.dumps([
        {"time_window": "07:00-08:00", "content": "起床 看一会儿天文新闻", "goal": "保持手感"},
        {"time_window": "09:00-12:00", "content": "跑光变曲线第三组分析", "goal": "M31 paper"},
        {"time_window": "14:00-18:00", "content": "写 paper 第二节"},
        {"time_window": "20:00-22:00", "content": "读 Rao 那篇预印本"},
    ])
    llm = FakeLLM(plan_json)
    store = FactStore(tmp_path / "facts.db")
    await store.init()
    life_writer = LifeWriter(store, scorer=None)
    planner = DailyPlanner(llm, FactRetriever(store), life_writer)

    await planner.plan_aria()

    plans = await store.query(subject="aria", type=FactType.PLAN, limit=10)
    assert len(plans) == 4
    assert all(p.importance == 7 for p in plans)
    assert all(any(t.startswith("time_window:") for t in p.tags) for p in plans)
    assert all(p.expires_at is not None for p in plans)


@pytest.mark.asyncio
async def test_plan_aria_uses_first_person_system(tmp_path):
    from lingxi.facts.store import FactStore
    from lingxi.facts.retriever import FactRetriever
    from lingxi.facts.writers.life import LifeWriter
    from lingxi.planner.daily_planner import DailyPlanner

    llm = FakeLLM(json.dumps([{"time_window": "09:00-10:00", "content": "x"}]))
    store = FactStore(tmp_path / "facts.db")
    await store.init()
    planner = DailyPlanner(llm, FactRetriever(store), LifeWriter(store, scorer=None))
    await planner.plan_aria()
    assert "你是 Aria" in llm.systems[0]
    assert "她" not in llm.calls[0]


@pytest.mark.asyncio
async def test_plan_aria_handles_llm_failure_gracefully(tmp_path):
    from lingxi.facts.store import FactStore
    from lingxi.facts.retriever import FactRetriever
    from lingxi.facts.writers.life import LifeWriter
    from lingxi.planner.daily_planner import DailyPlanner

    class BrokenLLM:
        async def complete(self, **kw):
            raise RuntimeError("api down")

    store = FactStore(tmp_path / "facts.db")
    await store.init()
    planner = DailyPlanner(BrokenLLM(), FactRetriever(store), LifeWriter(store, scorer=None))
    # Must not raise
    await planner.plan_aria()
    plans = await store.query(subject="aria", type=FactType.PLAN, limit=10)
    assert len(plans) == 0
