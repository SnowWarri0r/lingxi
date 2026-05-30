"""Tests for social.promoter — new facts-based implementation."""

from datetime import datetime, timedelta

import pytest

from lingxi.facts.store import FactStore
from lingxi.facts.writers.life import LifeWriter
from lingxi.facts.models import FactType
from lingxi.social.models import NPC, NPCEvent


def make_npc(id="xiaomin", name="小敏", relation="室友", background="室友，研究生"):
    return NPC(id=id, name=name, relation=relation, background=background)


def make_event(**overrides):
    defaults = dict(
        npc_id="xiaomin",
        ts=datetime.now() - timedelta(minutes=5),
        type="aria_interaction",
        content="拉 Aria 一起吃了麻辣烫，吵了几句关于学校食堂",
        significance=0.7,
    )
    defaults.update(overrides)
    return NPCEvent(**defaults)


@pytest.mark.asyncio
async def test_promoter_writes_fact_and_queues_intent(tmp_path):
    from lingxi.social.promoter import SocialPromoter
    from lingxi.proactive.share_intent import ShareIntentStore

    store = FactStore(tmp_path / "facts.db")
    await store.init()
    life_writer = LifeWriter(store)
    intent_store = ShareIntentStore(tmp_path)
    promoter = SocialPromoter(life_writer, intent_store, threshold=0.6)

    npc = make_npc()
    event = NPCEvent(
        npc_id="xiaomin",
        type="aria_interaction",
        content="Aria 给我加油打气了",
        significance=0.7,
        ts=datetime.now(),
        promoted_to_aria=False,
    )
    promoted = await promoter.maybe_promote(npc, event)
    assert promoted is True

    facts = await store.query(subject="aria", type=FactType.EVENT, limit=5)
    assert len(facts) == 1
    assert "小敏" in facts[0].content or "你" in facts[0].content
    assert any(t.startswith("from_npc:xiaomin") for t in facts[0].tags)

    intents = await intent_store.pending()
    assert len(intents) == 1
    assert intents[0].fact_id == facts[0].id


@pytest.mark.asyncio
async def test_promoter_respects_cooldown(tmp_path):
    from lingxi.social.promoter import SocialPromoter
    from lingxi.proactive.share_intent import ShareIntentStore

    store = FactStore(tmp_path / "facts.db")
    await store.init()
    promoter = SocialPromoter(
        LifeWriter(store),
        ShareIntentStore(tmp_path),
        threshold=0.6,
    )
    npc = make_npc()
    e1 = NPCEvent(npc_id="xiaomin", type="aria_interaction", content="Aria a",
                  significance=0.7, ts=datetime.now(), promoted_to_aria=False)
    e2 = NPCEvent(npc_id="xiaomin", type="aria_interaction", content="Aria b",
                  significance=0.7, ts=datetime.now(), promoted_to_aria=False)
    assert await promoter.maybe_promote(npc, e1) is True
    assert await promoter.maybe_promote(npc, e2) is False


@pytest.mark.asyncio
async def test_promoter_below_threshold_skipped(tmp_path):
    from lingxi.social.promoter import SocialPromoter
    from lingxi.proactive.share_intent import ShareIntentStore

    store = FactStore(tmp_path / "facts.db")
    await store.init()
    promoter = SocialPromoter(LifeWriter(store), ShareIntentStore(tmp_path), threshold=0.6)
    ev = make_event(significance=0.4)
    promoted = await promoter.maybe_promote(make_npc(), ev)
    assert promoted is False

    facts = await store.query(subject="aria", type=FactType.EVENT, limit=5)
    assert len(facts) == 0


@pytest.mark.asyncio
async def test_promoter_life_events_never_promoted(tmp_path):
    """NPC's own-life events should stay as background knowledge."""
    from lingxi.social.promoter import SocialPromoter
    from lingxi.proactive.share_intent import ShareIntentStore

    store = FactStore(tmp_path / "facts.db")
    await store.init()
    promoter = SocialPromoter(LifeWriter(store), ShareIntentStore(tmp_path), threshold=0.6)
    ev = make_event(type="life", significance=0.9, content="Echo 桌上发现一杯咖啡")
    promoted = await promoter.maybe_promote(make_npc(), ev)
    assert promoted is False

    facts = await store.query(subject="aria", type=FactType.EVENT, limit=5)
    assert len(facts) == 0


@pytest.mark.asyncio
async def test_promoter_already_promoted_flag_skipped(tmp_path):
    from lingxi.social.promoter import SocialPromoter
    from lingxi.proactive.share_intent import ShareIntentStore

    store = FactStore(tmp_path / "facts.db")
    await store.init()
    promoter = SocialPromoter(LifeWriter(store), ShareIntentStore(tmp_path), threshold=0.6)
    ev = make_event(promoted_to_aria=True)
    promoted = await promoter.maybe_promote(make_npc(), ev)
    assert promoted is False


@pytest.mark.asyncio
async def test_promoter_other_npc_not_blocked_by_cooldown(tmp_path):
    from lingxi.social.promoter import SocialPromoter
    from lingxi.proactive.share_intent import ShareIntentStore

    store = FactStore(tmp_path / "facts.db")
    await store.init()
    promoter = SocialPromoter(LifeWriter(store), ShareIntentStore(tmp_path), threshold=0.6)

    npc1 = make_npc(id="xiaomin", name="小敏")
    ev1 = make_event(npc_id="xiaomin", content="Aria x")
    assert await promoter.maybe_promote(npc1, ev1) is True

    npc2 = make_npc(id="tom", name="Tom")
    ev2 = make_event(npc_id="tom", content="Aria y")
    assert await promoter.maybe_promote(npc2, ev2) is True

    facts = await store.query(subject="aria", type=FactType.EVENT, limit=10)
    assert len(facts) == 2


@pytest.mark.asyncio
async def test_promoter_format_swaps_aria_to_niren(tmp_path):
    """Content with 'Aria' should be swapped to '你'."""
    from lingxi.social.promoter import SocialPromoter, _format_for_aria_pov
    from lingxi.proactive.share_intent import ShareIntentStore

    npc = make_npc(name="小敏")
    ev = make_event(
        type="aria_interaction",
        content="小敏拉 Aria 一起去吃了麻辣烫",
    )
    out = _format_for_aria_pov(npc, ev)
    assert "Aria" not in out
    assert "你" in out


def test_format_for_aria_pov_no_aria_fallback():
    from lingxi.social.promoter import _format_for_aria_pov

    npc = make_npc(name="小敏")
    ev = make_event(type="aria_interaction", content="拉着一起去看了电影")
    out = _format_for_aria_pov(npc, ev)
    assert out.startswith("小敏")
