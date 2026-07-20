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
async def test_executor_writes_event_for_current_plan(tmp_path):
    from lingxi.facts.store import FactStore
    from lingxi.facts.retriever import FactRetriever
    from lingxi.facts.writers.life import LifeWriter
    from lingxi.planner.executor import PlanExecutor

    store = FactStore(tmp_path / "facts.db")
    await store.init()
    life_writer = LifeWriter(store, scorer=None)
    now = datetime.now()
    plan = Fact(
        subject="aria", content="跑光变曲线第三组分析", source=Source.LIFE_SIMULATED,
        type=FactType.PLAN, ts=now, importance=7,
        expires_at=now.replace(hour=23, minute=59),
        tags=[f"time_window:{now.hour:02d}:00-{(now.hour+2)%24:02d}:00"],
    )
    await store.write(plan)

    llm = FakeLLM("第三组结果跑出来了，左侧有个小尖峰，正在排查是不是热噪声。")
    executor = PlanExecutor(llm, FactRetriever(store), life_writer)
    await executor.tick()

    events = await store.query(subject="aria", type=FactType.EVENT, limit=5)
    assert len(events) >= 1
    assert "尖峰" in events[0].content


@pytest.mark.asyncio
async def test_executor_skips_when_no_current_plan(tmp_path):
    from lingxi.facts.store import FactStore
    from lingxi.facts.retriever import FactRetriever
    from lingxi.facts.writers.life import LifeWriter
    from lingxi.planner.executor import PlanExecutor

    store = FactStore(tmp_path / "facts.db")
    await store.init()
    llm = FakeLLM("should not be called")
    executor = PlanExecutor(llm, FactRetriever(store), LifeWriter(store, scorer=None))
    await executor.tick()
    assert llm.calls == []


@pytest.mark.asyncio
async def test_executor_uses_first_person_aria_system(tmp_path):
    from lingxi.facts.store import FactStore
    from lingxi.facts.retriever import FactRetriever
    from lingxi.facts.writers.life import LifeWriter
    from lingxi.planner.executor import PlanExecutor

    store = FactStore(tmp_path / "facts.db")
    await store.init()
    now = datetime.now()
    await store.write(Fact(
        subject="aria", content="工作", source=Source.LIFE_SIMULATED,
        type=FactType.PLAN, ts=now, importance=7,
        tags=[f"time_window:{now.hour:02d}:00-{(now.hour+1)%24:02d}:00"],
    ))
    llm = FakeLLM("moment")
    executor = PlanExecutor(llm, FactRetriever(store), LifeWriter(store, scorer=None))
    await executor.tick()
    assert "你是 Aria" in llm.systems[0]
    assert "她" not in llm.calls[0]


@pytest.mark.asyncio
async def test_executor_request_replan_calls_planner_on_next_tick(tmp_path):
    from lingxi.facts.store import FactStore
    from lingxi.facts.retriever import FactRetriever
    from lingxi.facts.writers.life import LifeWriter
    from lingxi.planner.executor import PlanExecutor

    store = FactStore(tmp_path / "facts.db")
    await store.init()
    life_writer = LifeWriter(store, scorer=None)
    now = datetime.now()
    await store.write(Fact(
        subject="aria", content="work", source=Source.LIFE_SIMULATED,
        type=FactType.PLAN, ts=now, importance=7,
        tags=[f"time_window:{now.hour:02d}:00-{(now.hour+1)%24:02d}:00"],
    ))

    replan_calls = []
    class StubPlanner:
        async def plan_aria(self):
            replan_calls.append(True)

    llm = FakeLLM("moment")
    executor = PlanExecutor(
        llm, FactRetriever(store), life_writer, planner=StubPlanner()
    )
    executor.request_replan()
    await executor.tick()
    assert len(replan_calls) == 1


def test_time_window_minutes_and_midnight_crossing():
    from lingxi.planner.executor import _parse_time_window, _in_window

    # minutes precision: 13:30-16:00 excludes 13:00, includes 13:45
    w = _parse_time_window("13:30-16:00")
    assert w == (13 * 60 + 30, 16 * 60)
    assert not _in_window(13 * 60, *w)
    assert _in_window(13 * 60 + 45, *w)

    # same-hour window fires
    w = _parse_time_window("18:00-18:30")
    assert _in_window(18 * 60 + 10, *w)

    # midnight crossing: 23:00-01:00 covers 23:30 and 00:30, excludes 12:00
    w = _parse_time_window("23:00-01:00")
    assert _in_window(23 * 60 + 30, *w)
    assert _in_window(30, *w)
    assert not _in_window(12 * 60, *w)
