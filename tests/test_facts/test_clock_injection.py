from datetime import datetime, timedelta
from pathlib import Path

import pytest

from lingxi.facts.models import Fact, FactType, Source
from lingxi.facts.retriever import FactQuery, FactRetriever
from lingxi.facts.store import FactStore


FROZEN = datetime(2026, 8, 19, 20, 20, 54)


async def _store_with_two_facts(tmp_path: Path) -> FactStore:
    store = FactStore(tmp_path / "f.db")
    await store.init()
    # Same importance; only age separates them.
    await store.write(Fact(
        subject="aria", content="两小时前的事", source=Source.LIFE_SIMULATED,
        type=FactType.EVENT, ts=FROZEN - timedelta(hours=2), importance=5,
    ))
    await store.write(Fact(
        subject="aria", content="三十天前的事", source=Source.LIFE_SIMULATED,
        type=FactType.EVENT, ts=FROZEN - timedelta(days=30), importance=5,
    ))
    return store


@pytest.mark.asyncio
async def test_recency_ranking_uses_injected_clock(tmp_path):
    """Frozen at FROZEN, the 2-hour-old fact must outrank the 30-day-old one.

    Without injection the retriever scores against the real clock, so both
    facts age together as the case sits on disk and the ranking they encode
    silently drifts.
    """
    store = await _store_with_two_facts(tmp_path)
    retr = FactRetriever(store, clock=lambda: FROZEN)
    out = await retr.fetch(FactQuery(subject="aria", type=FactType.EVENT, limit=2))
    assert out[0].content == "两小时前的事"


@pytest.mark.asyncio
async def test_injected_clock_is_stable_across_calls(tmp_path):
    """Two fetches at the same frozen clock return the same order."""
    store = await _store_with_two_facts(tmp_path)
    retr = FactRetriever(store, clock=lambda: FROZEN)
    first = [f.content for f in await retr.fetch(
        FactQuery(subject="aria", type=FactType.EVENT, limit=2))]
    second = [f.content for f in await retr.fetch(
        FactQuery(subject="aria", type=FactType.EVENT, limit=2))]
    assert first == second


@pytest.mark.asyncio
async def test_default_clock_is_wall_clock(tmp_path):
    """Production path unchanged: no clock argument means datetime.now()."""
    store = FactStore(tmp_path / "f.db")
    await store.init()
    now = datetime.now()
    await store.write(Fact(
        subject="aria", content="刚刚", source=Source.LIFE_SIMULATED,
        type=FactType.EVENT, ts=now, importance=5,
    ))
    retr = FactRetriever(store)
    out = await retr.fetch(FactQuery(subject="aria", type=FactType.EVENT, limit=1))
    assert out[0].content == "刚刚"


@pytest.mark.asyncio
async def test_store_expiry_uses_injected_clock(tmp_path):
    """A fact that expired before the frozen clock is filtered out."""
    store = FactStore(tmp_path / "f.db", clock=lambda: FROZEN)
    await store.init()
    await store.write(Fact(
        subject="aria", content="已过期", source=Source.LIFE_SIMULATED,
        type=FactType.EVENT, ts=FROZEN - timedelta(days=2), importance=5,
        expires_at=FROZEN - timedelta(days=1),
    ))
    await store.write(Fact(
        subject="aria", content="没过期", source=Source.LIFE_SIMULATED,
        type=FactType.EVENT, ts=FROZEN - timedelta(days=2), importance=5,
        expires_at=FROZEN + timedelta(days=1),
    ))
    rows = await store.query(subject="aria", limit=10)
    assert [r.content for r in rows] == ["没过期"]
