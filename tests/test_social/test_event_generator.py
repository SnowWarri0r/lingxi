"""Tests for social.event_generator."""

import json
import random
from datetime import datetime, timedelta

import pytest

from lingxi.social.event_generator import (
    compute_tick_probability,
    generate_events,
    should_tick,
)
from lingxi.social.models import NPC, NPCArc, NPCEvent, NPCState


def make_npc(**overrides) -> NPC:
    defaults = dict(
        id="xiaomin",
        name="小敏",
        relation="室友",
        background="同所博士",
        base_event_probability=0.3,
    )
    defaults.update(overrides)
    return NPC(**defaults)


def make_state(**overrides) -> NPCState:
    return NPCState(npc_id="xiaomin", **overrides)


class TestComputeTickProbability:
    def test_base_only(self):
        npc = make_npc(base_event_probability=0.3)
        state = make_state()
        # No last event AND no high-weight arc → +0.2 catch-up still applies
        # (last_event_at is None counts as "more than 24h ago")
        p = compute_tick_probability(npc, state, now=datetime(2026, 5, 22, 12))
        assert p == pytest.approx(0.5)

    def test_recent_event_no_catchup_bonus(self):
        npc = make_npc(base_event_probability=0.3)
        state = make_state(last_event_at=datetime(2026, 5, 22, 10))
        p = compute_tick_probability(npc, state, now=datetime(2026, 5, 22, 12))
        # last_event_at within 24h → no +0.2 catch-up, no arc → 0.3
        assert p == pytest.approx(0.3)

    def test_high_weight_arc_boost(self):
        npc = make_npc(base_event_probability=0.3)
        state = make_state(
            last_event_at=datetime(2026, 5, 22, 10),
            arcs=[NPCArc(id="a", summary="x", stage="mid", weight=0.8)],
        )
        p = compute_tick_probability(npc, state, now=datetime(2026, 5, 22, 12))
        # 0.3 base + 0.2 arc-weight, no catch-up
        assert p == pytest.approx(0.5)

    def test_late_evening_dampens(self):
        npc = make_npc(base_event_probability=0.5)
        state = make_state(last_event_at=datetime(2026, 5, 22, 20))
        p = compute_tick_probability(npc, state, now=datetime(2026, 5, 22, 22))
        # 0.5 base * 0.3 late = 0.15
        assert p == pytest.approx(0.15)

    def test_clamped_to_one(self):
        npc = make_npc(base_event_probability=0.9)
        state = make_state(
            arcs=[NPCArc(id="a", summary="x", stage="mid", weight=0.9)],
        )
        # 0.9 + 0.2 catch-up + 0.2 arc = 1.3, clamped
        p = compute_tick_probability(npc, state, now=datetime(2026, 5, 22, 12))
        assert p == 1.0

    def test_resolved_arcs_dont_boost(self):
        npc = make_npc(base_event_probability=0.3)
        state = make_state(
            last_event_at=datetime(2026, 5, 22, 10),
            arcs=[NPCArc(id="a", summary="x", stage="resolved", weight=0.9)],
        )
        p = compute_tick_probability(npc, state, now=datetime(2026, 5, 22, 12))
        # resolved arc doesn't trigger weight boost
        assert p == pytest.approx(0.3)


class TestShouldTick:
    def test_deterministic_high(self):
        rng = random.Random(42)
        # With p=1.0, always fires
        assert should_tick(1.0, rng=rng) is True

    def test_deterministic_low(self):
        rng = random.Random(42)
        assert should_tick(0.0, rng=rng) is False


# --- generate_events with a fake LLM ---


class FakeLLMResponse:
    def __init__(self, content: str):
        self.content = content


class FakeLLM:
    def __init__(self, response_text: str):
        self.response_text = response_text
        self.calls = []

    async def complete(self, **kwargs):
        self.calls.append(kwargs)
        return FakeLLMResponse(self.response_text)


class TestGenerateEvents:
    @pytest.mark.asyncio
    async def test_basic_parse(self):
        payload = json.dumps([
            {"type": "life", "content": "点了麦当劳", "significance": 0.2, "arc_id": None},
            {"type": "aria_interaction", "content": "拉 Aria 去吃饭",
             "significance": 0.4, "arc_id": "thesis_pressure"},
        ])
        llm = FakeLLM(payload)
        npc = make_npc()
        state = make_state(arcs=[NPCArc(id="thesis_pressure", summary="x", stage="early")])
        events = await generate_events(llm, npc, state, now=datetime(2026, 5, 22, 14))
        assert len(events) == 2
        assert events[0].type == "life"
        assert events[0].content == "点了麦当劳"
        assert events[1].type == "aria_interaction"
        assert events[1].arc_id == "thesis_pressure"

    @pytest.mark.asyncio
    async def test_caps_at_two_events(self):
        payload = json.dumps([
            {"type": "life", "content": "事1", "significance": 0.2},
            {"type": "life", "content": "事2", "significance": 0.2},
            {"type": "life", "content": "事3", "significance": 0.2},
            {"type": "life", "content": "事4", "significance": 0.2},
        ])
        llm = FakeLLM(payload)
        events = await generate_events(llm, make_npc(), make_state())
        assert len(events) == 2

    @pytest.mark.asyncio
    async def test_clamps_significance(self):
        payload = json.dumps([
            {"type": "life", "content": "x", "significance": 2.5},
        ])
        llm = FakeLLM(payload)
        events = await generate_events(llm, make_npc(), make_state())
        assert events[0].significance == 1.0

    @pytest.mark.asyncio
    async def test_drops_unknown_arc_id(self):
        payload = json.dumps([
            {"type": "life", "content": "x", "significance": 0.3,
             "arc_id": "made_up_arc"},
        ])
        llm = FakeLLM(payload)
        state = make_state(arcs=[NPCArc(id="real_arc", summary="x", stage="early")])
        events = await generate_events(llm, make_npc(), state)
        assert events[0].arc_id is None

    @pytest.mark.asyncio
    async def test_keeps_valid_arc_id(self):
        payload = json.dumps([
            {"type": "life", "content": "x", "significance": 0.3,
             "arc_id": "real_arc"},
        ])
        llm = FakeLLM(payload)
        state = make_state(arcs=[NPCArc(id="real_arc", summary="x", stage="early")])
        events = await generate_events(llm, make_npc(), state)
        assert events[0].arc_id == "real_arc"

    @pytest.mark.asyncio
    async def test_invalid_type_falls_back_to_life(self):
        payload = json.dumps([
            {"type": "bogus", "content": "x", "significance": 0.3},
        ])
        llm = FakeLLM(payload)
        events = await generate_events(llm, make_npc(), make_state())
        assert events[0].type == "life"

    @pytest.mark.asyncio
    async def test_empty_content_skipped(self):
        payload = json.dumps([
            {"type": "life", "content": "", "significance": 0.3},
            {"type": "life", "content": "real", "significance": 0.3},
        ])
        llm = FakeLLM(payload)
        events = await generate_events(llm, make_npc(), make_state())
        assert len(events) == 1
        assert events[0].content == "real"

    @pytest.mark.asyncio
    async def test_llm_garbage_returns_empty(self):
        llm = FakeLLM("not json at all, just text")
        events = await generate_events(llm, make_npc(), make_state())
        assert events == []

    @pytest.mark.asyncio
    async def test_markdown_fence_stripped(self):
        payload = (
            "```json\n"
            + json.dumps([{"type": "life", "content": "x", "significance": 0.3}])
            + "\n```"
        )
        llm = FakeLLM(payload)
        events = await generate_events(llm, make_npc(), make_state())
        assert len(events) == 1
        assert events[0].content == "x"

    @pytest.mark.asyncio
    async def test_prompt_includes_recent_events(self):
        llm = FakeLLM("[]")
        state = make_state(
            recent_events=[NPCEvent(
                npc_id="xiaomin",
                ts=datetime.now() - timedelta(hours=3),
                type="life",
                content="昨天点了外卖",
                significance=0.2,
            )]
        )
        await generate_events(llm, make_npc(), state)
        prompt = llm.calls[0]["messages"][0]["content"]
        assert "昨天点了外卖" in prompt

    @pytest.mark.asyncio
    async def test_prompt_includes_active_arcs(self):
        llm = FakeLLM("[]")
        state = make_state(
            arcs=[
                NPCArc(id="a", summary="论文压力", stage="mid", weight=0.8),
                NPCArc(id="b", summary="老事", stage="resolved", weight=0.5),
            ]
        )
        await generate_events(llm, make_npc(), state)
        prompt = llm.calls[0]["messages"][0]["content"]
        assert "论文压力" in prompt
        assert "老事" not in prompt  # resolved arcs hidden
