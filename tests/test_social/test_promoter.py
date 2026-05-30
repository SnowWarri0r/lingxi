"""Tests for social.promoter."""

from datetime import datetime, timedelta

import pytest

from lingxi.inner_life.store import InnerLifeStore
from lingxi.social.models import NPC, NPCEvent
from lingxi.social.promoter import SocialPromoter, _format_for_inner_state
from lingxi.social.store import SocialStore


def make_npc(id="xiaomin", name="小敏", relation="室友"):
    return NPC(id=id, name=name, relation=relation, background="x")


def make_event(**overrides):
    defaults = dict(
        npc_id="xiaomin",
        ts=datetime.now() - timedelta(minutes=5),
        type="life",
        content="跟导师吵了一架",
        significance=0.7,
    )
    defaults.update(overrides)
    return NPCEvent(**defaults)


@pytest.fixture
def stores(tmp_path):
    inner = InnerLifeStore(tmp_path / "inner")
    social = SocialStore(tmp_path)
    return inner, social, tmp_path


class TestThreshold:
    @pytest.mark.asyncio
    async def test_below_threshold_skipped(self, stores):
        inner, social, root = stores
        promoter = SocialPromoter(inner, social, root)
        ev = make_event(significance=0.4)
        promoted = await promoter.maybe_promote(make_npc(), ev)
        assert promoted is False
        state = await inner.load_state()
        assert state.recent_events == []

    @pytest.mark.asyncio
    async def test_above_threshold_promoted(self, stores):
        inner, social, root = stores
        promoter = SocialPromoter(inner, social, root)
        ev = make_event(significance=0.7)
        # Need to actually persist the event first so mark_event_promoted has
        # something to find. The scheduler appends before calling the hook,
        # so we mirror that here.
        await social.append_event(ev)
        promoted = await promoter.maybe_promote(make_npc(), ev)
        assert promoted is True

        state = await inner.load_state()
        assert len(state.recent_events) == 1
        assert state.recent_events[0].wants_to_share is True
        assert "小敏" in state.recent_events[0].content

    @pytest.mark.asyncio
    async def test_already_promoted_skipped(self, stores):
        inner, social, root = stores
        promoter = SocialPromoter(inner, social, root)
        ev = make_event(promoted_to_aria=True)
        promoted = await promoter.maybe_promote(make_npc(), ev)
        assert promoted is False


class TestCooldown:
    @pytest.mark.asyncio
    async def test_second_push_within_24h_blocked(self, stores):
        inner, social, root = stores
        promoter = SocialPromoter(inner, social, root)
        npc = make_npc()

        ev1 = make_event(content="事1", ts=datetime.now() - timedelta(hours=2))
        await social.append_event(ev1)
        assert await promoter.maybe_promote(npc, ev1) is True

        ev2 = make_event(content="事2", ts=datetime.now())
        await social.append_event(ev2)
        assert await promoter.maybe_promote(npc, ev2) is False

        state = await inner.load_state()
        assert len(state.recent_events) == 1

    @pytest.mark.asyncio
    async def test_other_npc_not_blocked_by_cooldown(self, stores):
        inner, social, root = stores
        promoter = SocialPromoter(inner, social, root)

        npc1 = make_npc(id="xiaomin", name="小敏")
        ev1 = make_event(npc_id="xiaomin", content="x")
        await social.append_event(ev1)
        assert await promoter.maybe_promote(npc1, ev1) is True

        npc2 = make_npc(id="tom", name="Tom")
        ev2 = make_event(npc_id="tom", content="y")
        await social.append_event(ev2)
        assert await promoter.maybe_promote(npc2, ev2) is True

        state = await inner.load_state()
        assert len(state.recent_events) == 2

    @pytest.mark.asyncio
    async def test_short_cooldown_allows_repush(self, stores):
        inner, social, root = stores
        promoter = SocialPromoter(
            inner, social, root, cooldown=timedelta(milliseconds=1),
        )
        npc = make_npc()
        ev1 = make_event(content="x", ts=datetime.now() - timedelta(seconds=10))
        await social.append_event(ev1)
        assert await promoter.maybe_promote(npc, ev1) is True

        # Tiny sleep to exceed cooldown
        import asyncio
        await asyncio.sleep(0.01)

        ev2 = make_event(content="y", ts=datetime.now())
        await social.append_event(ev2)
        assert await promoter.maybe_promote(npc, ev2) is True


class TestFormat:
    def test_life_event_prefixes_name(self):
        npc = make_npc(name="小敏")
        ev = make_event(type="life", content="跟导师吵了一架")
        out = _format_for_inner_state(npc, ev)
        assert out == "小敏跟导师吵了一架"

    def test_life_event_already_prefixed_not_double_prefixed(self):
        npc = make_npc(name="小敏")
        ev = make_event(type="life", content="小敏改 paper 到凌晨")
        out = _format_for_inner_state(npc, ev)
        assert out == "小敏改 paper 到凌晨"
        assert "小敏小敏" not in out

    def test_aria_interaction_swaps_to_second_person(self):
        npc = make_npc(name="小敏")
        ev = make_event(
            type="aria_interaction",
            content="小敏拉 Aria 一起去吃了麻辣烫",
        )
        out = _format_for_inner_state(npc, ev)
        assert "Aria" not in out
        assert "你" in out

    def test_aria_interaction_without_aria_string_falls_back(self):
        npc = make_npc(name="小敏")
        ev = make_event(
            type="aria_interaction",
            content="拉着一起去看了电影",  # no "Aria" literal
        )
        out = _format_for_inner_state(npc, ev)
        assert out.startswith("小敏")


class TestMarkPromoted:
    @pytest.mark.asyncio
    async def test_npc_event_marked_promoted_after_push(self, stores):
        inner, social, root = stores
        promoter = SocialPromoter(inner, social, root)
        ev = make_event()
        await social.append_event(ev)
        await promoter.maybe_promote(make_npc(), ev)
        state = await social.load_state("xiaomin")
        assert state.recent_events[0].promoted_to_aria is True
