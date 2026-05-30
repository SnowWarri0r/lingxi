"""Tests for social.scheduler (cron tick logic, idempotency)."""

import json
import random
from datetime import datetime, timedelta

import pytest

from lingxi.social.models import NPC, NPCArc, SocialGraph
from lingxi.social.scheduler import DEFAULT_TICK_HOURS, SocialScheduler
from lingxi.social.store import SocialStore


class FakeLLMResponse:
    def __init__(self, content: str):
        self.content = content


class FakeLLM:
    def __init__(self, payload: list[dict] | None = None):
        self._payload = payload or [
            {"type": "life", "content": "fake event", "significance": 0.3,
             "arc_id": None}
        ]
        self.call_count = 0

    async def complete(self, **kwargs):
        self.call_count += 1
        return FakeLLMResponse(json.dumps(self._payload))


@pytest.fixture
def store(tmp_path):
    return SocialStore(tmp_path)


@pytest.fixture
def graph():
    return SocialGraph(npcs=[
        NPC(
            id="xiaomin", name="小敏", relation="室友",
            background="x", base_event_probability=0.5,
            initial_arcs=[NPCArc(id="thesis", summary="论文压力", weight=0.8)],
        ),
        NPC(
            id="tom", name="Tom", relation="同事",
            background="y", base_event_probability=0.3,
        ),
    ])


class TestTrigger:
    @pytest.mark.asyncio
    async def test_trigger_now_runs_for_each_npc(self, store, graph):
        # Seed arcs so xiaomin's thesis arc id is recognized
        await store.save_arcs("xiaomin", list(graph.npcs[0].initial_arcs))

        llm = FakeLLM()
        # Deterministic rng → always rolls < 1, both NPCs tick
        rng = random.Random(0)
        # Force probability to 1 by manipulating npc base
        for npc in graph.npcs:
            npc.base_event_probability = 1.0

        sched = SocialScheduler(llm, graph, store, rng=rng)
        await sched.trigger_now()

        # store.append_event removed in P7 (dual-write drop).
        # Events now go to npc_writer (facts table) only.
        # Verify both NPCs got an LLM event-generation call.
        assert llm.call_count == 2

    @pytest.mark.asyncio
    async def test_trigger_writes_last_tick(self, store, graph):
        llm = FakeLLM()
        for npc in graph.npcs:
            npc.base_event_probability = 0.0  # No events generated
        sched = SocialScheduler(llm, graph, store)
        await sched.trigger_now()
        last = await store.load_last_tick()
        assert last is not None


class TestProbabilityGate:
    @pytest.mark.asyncio
    async def test_low_probability_skips_npc(self, store, graph):
        llm = FakeLLM()
        for npc in graph.npcs:
            npc.base_event_probability = 0.0
        # Even with catch-up bonus, p = 0 + 0.2 = 0.2
        # Seed rng so it returns > 0.2
        rng = random.Random()
        rng.random = lambda: 0.99  # always over threshold
        sched = SocialScheduler(llm, graph, store, rng=rng)
        await sched.trigger_now()
        # No LLM calls because dice failed for both
        assert llm.call_count == 0


class TestArcCountBump:
    @pytest.mark.asyncio
    async def test_arc_event_count_increments(self, store, graph):
        await store.save_arcs("xiaomin", list(graph.npcs[0].initial_arcs))
        llm = FakeLLM(payload=[
            {"type": "life", "content": "改 paper", "significance": 0.4,
             "arc_id": "thesis"},
        ])
        for npc in graph.npcs:
            npc.base_event_probability = 1.0
        rng = random.Random()
        rng.random = lambda: 0.0  # always fires
        sched = SocialScheduler(llm, graph, store, rng=rng)
        await sched.trigger_now()
        state = await store.load_state("xiaomin")
        thesis = next(a for a in state.arcs if a.id == "thesis")
        assert thesis.event_count == 1

    @pytest.mark.asyncio
    async def test_no_arc_id_no_bump(self, store, graph):
        await store.save_arcs("xiaomin", list(graph.npcs[0].initial_arcs))
        llm = FakeLLM(payload=[
            {"type": "life", "content": "事", "significance": 0.2, "arc_id": None},
        ])
        for npc in graph.npcs:
            npc.base_event_probability = 1.0
        rng = random.Random()
        rng.random = lambda: 0.0
        sched = SocialScheduler(llm, graph, store, rng=rng)
        await sched.trigger_now()
        state = await store.load_state("xiaomin")
        thesis = next(a for a in state.arcs if a.id == "thesis")
        assert thesis.event_count == 0


class TestEventHook:
    @pytest.mark.asyncio
    async def test_on_event_written_called(self, store, graph):
        llm = FakeLLM(payload=[
            {"type": "life", "content": "x", "significance": 0.7, "arc_id": None},
        ])
        for npc in graph.npcs:
            npc.base_event_probability = 1.0
        rng = random.Random()
        rng.random = lambda: 0.0

        seen = []

        async def hook(npc, event):
            seen.append((npc.id, event.significance))

        sched = SocialScheduler(llm, graph, store, rng=rng, on_event_written=hook)
        await sched.trigger_now()
        assert len(seen) == 2  # both NPCs
        assert all(npc_id in ("xiaomin", "tom") for npc_id, _ in seen)


class TestTickHourGate:
    """_maybe_tick should only fire during configured hours."""

    @pytest.mark.asyncio
    async def test_default_tick_hours_are_even_8_to_22(self):
        assert DEFAULT_TICK_HOURS == (8, 10, 12, 14, 16, 18, 20, 22)
