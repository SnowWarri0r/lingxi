"""Tests for social.models + social.store."""

from datetime import datetime, timedelta

import pytest

from lingxi.social.models import NPC, NPCArc, NPCEvent, NPCState, SocialGraph
from lingxi.social.store import SocialStore


@pytest.fixture
def store(tmp_path):
    return SocialStore(tmp_path)


class TestModels:
    def test_arc_defaults(self):
        a = NPCArc(id="x", summary="...")
        assert a.stage == "early"
        assert a.event_count == 0
        assert a.resolution is None

    def test_event_minimal(self):
        e = NPCEvent(
            npc_id="x",
            ts=datetime(2026, 5, 22, 10, 0),
            type="life",
            content="...",
            significance=0.3,
        )
        assert e.promoted_to_aria is False
        assert e.arc_id is None

    def test_npc_state_active_arcs_filters_resolved(self):
        state = NPCState(
            npc_id="x",
            arcs=[
                NPCArc(id="a", summary="x", stage="early"),
                NPCArc(id="b", summary="y", stage="resolved"),
                NPCArc(id="c", summary="z", stage="mid"),
            ],
        )
        active = state.active_arcs()
        assert len(active) == 2
        assert {a.id for a in active} == {"a", "c"}

    def test_graph_by_id_lookup(self):
        g = SocialGraph(npcs=[
            NPC(id="xiaomin", name="小敏", relation="室友", background="..."),
        ])
        assert g.by_id("xiaomin") is not None
        assert g.by_id("none") is None


class TestStore:
    @pytest.mark.asyncio
    async def test_load_missing_returns_empty(self, store):
        s = await store.load_state("nobody")
        assert s.npc_id == "nobody"
        assert s.arcs == []
        assert s.recent_events == []
        assert s.last_event_at is None

    @pytest.mark.asyncio
    async def test_append_and_load_event(self, store):
        ev = NPCEvent(
            npc_id="xiaomin",
            ts=datetime.now() - timedelta(hours=1),
            type="life",
            content="点了 10 点的外卖",
            significance=0.2,
        )
        await store.append_event(ev)
        state = await store.load_state("xiaomin")
        assert len(state.recent_events) == 1
        assert state.recent_events[0].content == "点了 10 点的外卖"
        assert state.last_event_at is not None

    @pytest.mark.asyncio
    async def test_events_trimmed_by_cutoff(self, store):
        old = NPCEvent(
            npc_id="x",
            ts=datetime.now() - timedelta(days=45),
            type="life",
            content="老事",
            significance=0.2,
        )
        new = NPCEvent(
            npc_id="x",
            ts=datetime.now() - timedelta(hours=2),
            type="life",
            content="新事",
            significance=0.2,
        )
        await store.append_event(old)
        await store.append_event(new)
        state = await store.load_state("x")  # default cutoff = 30 days
        assert len(state.recent_events) == 1
        assert state.recent_events[0].content == "新事"

    @pytest.mark.asyncio
    async def test_events_sorted_ascending(self, store):
        ts = datetime.now() - timedelta(hours=5)
        await store.append_event(NPCEvent(
            npc_id="x", ts=ts + timedelta(hours=3), type="life",
            content="第二", significance=0.2,
        ))
        await store.append_event(NPCEvent(
            npc_id="x", ts=ts, type="life",
            content="第一", significance=0.2,
        ))
        state = await store.load_state("x")
        assert [e.content for e in state.recent_events] == ["第一", "第二"]

    @pytest.mark.asyncio
    async def test_save_and_load_arcs(self, store):
        arcs = [
            NPCArc(id="a", summary="论文压力", stage="early", weight=0.8),
            NPCArc(id="b", summary="跟室友吵架", stage="mid", weight=0.4),
        ]
        await store.save_arcs("xiaomin", arcs)
        state = await store.load_state("xiaomin")
        assert len(state.arcs) == 2
        assert state.arcs[0].summary == "论文压力"

    @pytest.mark.asyncio
    async def test_mark_event_promoted(self, store):
        ts = datetime.now() - timedelta(hours=1)
        await store.append_event(NPCEvent(
            npc_id="x", ts=ts, type="life", content="大事", significance=0.7,
        ))
        await store.mark_event_promoted("x", ts)
        state = await store.load_state("x")
        assert state.recent_events[0].promoted_to_aria is True

    @pytest.mark.asyncio
    async def test_last_tick_roundtrip(self, store):
        assert await store.load_last_tick() is None
        ts = datetime(2026, 5, 22, 14, 0)
        await store.save_last_tick(ts)
        loaded = await store.load_last_tick()
        assert loaded == ts

    @pytest.mark.asyncio
    async def test_corrupted_events_file_skipped(self, tmp_path, store):
        path = tmp_path / "social" / "npcs" / "x" / "events.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        good = NPCEvent(
            npc_id="x", ts=datetime.now() - timedelta(hours=1),
            type="life", content="好的", significance=0.2,
        ).model_dump_json()
        path.write_text(f"{{ broken json\n{good}\n", encoding="utf-8")
        state = await store.load_state("x")
        assert len(state.recent_events) == 1
        assert state.recent_events[0].content == "好的"
