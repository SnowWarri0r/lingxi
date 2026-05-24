"""Tests for social.arc_advancer."""

import json
from datetime import datetime

import pytest

from lingxi.social.arc_advancer import (
    MAX_EVENTS_PER_ARC,
    STAGE_THRESHOLDS,
    advance_npc_arcs,
    maybe_advance_arc,
)
from lingxi.social.models import NPC, NPCArc, NPCEvent, NPCState
from lingxi.social.store import SocialStore


def make_npc(id="x"):
    return NPC(id=id, name="N", relation="室友", background="...")


def make_arc(**overrides):
    defaults = dict(id="a", summary="...", stage="early", event_count=0)
    defaults.update(overrides)
    return NPCArc(**defaults)


def make_state(arcs=None, events=None):
    return NPCState(npc_id="x", arcs=arcs or [], recent_events=events or [])


class FakeLLMResponse:
    def __init__(self, content): self.content = content


class FakeLLM:
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    async def complete(self, **kwargs):
        self.calls += 1
        return FakeLLMResponse(json.dumps(self.payload))


class TestThresholds:
    @pytest.mark.asyncio
    async def test_below_threshold_no_llm_call(self):
        llm = FakeLLM({"advance": True, "new_summary": "x"})
        arc = make_arc(stage="early", event_count=2)  # < 3
        result = await maybe_advance_arc(llm, make_npc(), arc, make_state())
        assert result is None
        assert llm.calls == 0

    @pytest.mark.asyncio
    async def test_at_threshold_calls_llm(self):
        llm = FakeLLM({"advance": True, "new_summary": "进展中"})
        arc = make_arc(stage="early", event_count=STAGE_THRESHOLDS["early"])
        result = await maybe_advance_arc(llm, make_npc(), arc, make_state())
        assert llm.calls == 1
        assert result is not None
        assert result.stage == "mid"

    @pytest.mark.asyncio
    async def test_resolved_arc_skipped(self):
        llm = FakeLLM({"advance": True})
        arc = make_arc(stage="resolved", event_count=100)
        result = await maybe_advance_arc(llm, make_npc(), arc, make_state())
        assert result is None
        assert llm.calls == 0


class TestForceResolve:
    @pytest.mark.asyncio
    async def test_excess_events_force_resolve_no_llm(self):
        llm = FakeLLM({"advance": False})
        arc = make_arc(stage="mid", event_count=MAX_EVENTS_PER_ARC + 1)
        result = await maybe_advance_arc(llm, make_npc(), arc, make_state())
        assert result is not None
        assert result.stage == "resolved"
        assert result.resolution is not None
        assert llm.calls == 0


class TestStageProgression:
    @pytest.mark.asyncio
    async def test_early_to_mid(self):
        llm = FakeLLM({"advance": True, "new_summary": "中段了"})
        arc = make_arc(stage="early", event_count=STAGE_THRESHOLDS["early"])
        result = await maybe_advance_arc(llm, make_npc(), arc, make_state())
        assert result.stage == "mid"
        assert result.summary == "中段了"
        assert result.event_count == 0  # reset on advance

    @pytest.mark.asyncio
    async def test_mid_to_climax(self):
        llm = FakeLLM({"advance": True, "new_summary": "到关键节点"})
        arc = make_arc(stage="mid", event_count=STAGE_THRESHOLDS["mid"], weight=0.5)
        result = await maybe_advance_arc(llm, make_npc(), arc, make_state())
        assert result.stage == "climax"
        # climax bumps weight
        assert result.weight == 0.6

    @pytest.mark.asyncio
    async def test_climax_to_resolved_via_chain(self):
        llm = FakeLLM({"advance": True, "new_summary": "结束"})
        arc = make_arc(stage="climax", event_count=STAGE_THRESHOLDS["climax"])
        result = await maybe_advance_arc(llm, make_npc(), arc, make_state())
        assert result.stage == "resolved"

    @pytest.mark.asyncio
    async def test_resolution_short_circuits_to_resolved(self):
        # LLM says advance + provides resolution → skip directly to resolved
        # even from "early" stage
        llm = FakeLLM({
            "advance": True,
            "new_summary": "突然结束了",
            "resolution": "事情突然有了答案，arc 结束",
        })
        arc = make_arc(stage="early", event_count=STAGE_THRESHOLDS["early"])
        result = await maybe_advance_arc(llm, make_npc(), arc, make_state())
        assert result.stage == "resolved"
        assert result.resolution == "事情突然有了答案，arc 结束"

    @pytest.mark.asyncio
    async def test_llm_says_no_keeps_arc_as_is(self):
        llm = FakeLLM({"advance": False})
        arc = make_arc(stage="early", event_count=STAGE_THRESHOLDS["early"])
        result = await maybe_advance_arc(llm, make_npc(), arc, make_state())
        assert result is None


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_llm_garbage_returns_none(self):
        class BadLLM:
            async def complete(self, **kwargs):
                return FakeLLMResponse("not json")
        arc = make_arc(stage="early", event_count=STAGE_THRESHOLDS["early"])
        result = await maybe_advance_arc(BadLLM(), make_npc(), arc, make_state())
        assert result is None

    @pytest.mark.asyncio
    async def test_llm_exception_returns_none(self):
        class ExceptionLLM:
            async def complete(self, **kwargs):
                raise RuntimeError("boom")
        arc = make_arc(stage="early", event_count=STAGE_THRESHOLDS["early"])
        result = await maybe_advance_arc(ExceptionLLM(), make_npc(), arc, make_state())
        assert result is None


class TestAdvanceNpcArcs:
    @pytest.mark.asyncio
    async def test_persists_advance_to_store(self, tmp_path):
        store = SocialStore(tmp_path)
        npc = make_npc()
        await store.save_arcs(npc.id, [
            make_arc(id="a", stage="early",
                     event_count=STAGE_THRESHOLDS["early"]),
            make_arc(id="b", stage="early", event_count=1),  # below threshold
        ])
        llm = FakeLLM({"advance": True, "new_summary": "进展"})
        count = await advance_npc_arcs(llm, npc, store)
        assert count == 1
        state = await store.load_state(npc.id)
        arc_a = next(a for a in state.arcs if a.id == "a")
        arc_b = next(a for a in state.arcs if a.id == "b")
        assert arc_a.stage == "mid"
        assert arc_b.stage == "early"  # untouched

    @pytest.mark.asyncio
    async def test_no_change_no_persist_call(self, tmp_path):
        store = SocialStore(tmp_path)
        npc = make_npc()
        await store.save_arcs(npc.id, [
            make_arc(id="a", stage="early", event_count=1),
        ])
        llm = FakeLLM({"advance": True, "new_summary": "x"})
        count = await advance_npc_arcs(llm, npc, store)
        assert count == 0
