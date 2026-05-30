"""Tests for find_pending_share helper in proactive.py."""

import pytest
from datetime import datetime
from lingxi.facts.store import FactStore
from lingxi.facts.retriever import FactRetriever
from lingxi.facts.writers.life import LifeWriter
from lingxi.facts.models import Fact, Source, FactType
from lingxi.proactive.share_intent import ShareIntentStore


@pytest.mark.asyncio
async def test_find_pending_share_picks_highest_significance(tmp_path):
    """find_pending_share returns the highest-significance queued intent."""
    from lingxi.temporal.proactive import find_pending_share

    store = FactStore(tmp_path / "facts.db")
    await store.init()
    life_writer = LifeWriter(store)
    intent_store = ShareIntentStore(tmp_path)
    retriever = FactRetriever(store)

    f1 = Fact(subject="aria", content="低 sig 事件", source=Source.NPC_TICKER,
              type=FactType.EVENT, ts=datetime.now())
    f2 = Fact(subject="aria", content="高 sig 事件", source=Source.NPC_TICKER,
              type=FactType.EVENT, ts=datetime.now())
    await life_writer.write(f1)
    await life_writer.write(f2)
    await intent_store.queue(f1.id, "echo", 0.6)
    await intent_store.queue(f2.id, "xiaomin", 0.9)

    result = await find_pending_share(intent_store, retriever)
    assert result is not None
    intent, fact = result
    assert fact.id == f2.id


@pytest.mark.asyncio
async def test_find_pending_share_skips_stale(tmp_path):
    """If a queued intent references a missing fact, consume it and try next."""
    from lingxi.temporal.proactive import find_pending_share

    store = FactStore(tmp_path / "facts.db")
    await store.init()
    intent_store = ShareIntentStore(tmp_path)
    retriever = FactRetriever(store)

    # Queue an intent for a fact_id that doesn't exist
    await intent_store.queue("stale_id", "xiaomin", 0.9)
    result = await find_pending_share(intent_store, retriever)
    assert result is None
    # Stale intent was cleaned up
    pending = await intent_store.pending()
    assert pending == []


@pytest.mark.asyncio
async def test_find_pending_share_empty_returns_none(tmp_path):
    from lingxi.temporal.proactive import find_pending_share

    store = FactStore(tmp_path / "facts.db")
    await store.init()
    intent_store = ShareIntentStore(tmp_path)
    retriever = FactRetriever(store)
    assert await find_pending_share(intent_store, retriever) is None
