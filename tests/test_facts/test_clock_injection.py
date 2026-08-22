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


async def _store_with_recency_importance_tradeoff(tmp_path: Path) -> FactStore:
    """One recent-but-unimportant fact and one old-but-important fact.

    fetch() scores as 0.5*recency + 0.3*(importance/10) + 0.2*fts_rank, and
    recency = exp(-0.01 * hours_old). Ordering two facts by recency alone is
    monotonic in fact.ts for any fixed "now", so it can never tell whether the
    injected clock (vs. the real one) was actually used -- both clocks would
    produce the same order. To make the clock argument observable, these two
    facts are built so the *full* score flips depending on how far the clock
    sits from them: near the facts, the recency term dominates and the recent
    fact wins; far in the future, recency decays toward 0 for both and the
    fixed importance term decides, so the important-but-old fact wins.
    """
    store = FactStore(tmp_path / "f.db")
    await store.init()
    await store.write(Fact(
        subject="aria", content="最近但不重要", source=Source.LIFE_SIMULATED,
        type=FactType.EVENT, ts=FROZEN - timedelta(hours=1), importance=1,
    ))
    await store.write(Fact(
        subject="aria", content="久远但重要", source=Source.LIFE_SIMULATED,
        type=FactType.EVENT, ts=FROZEN - timedelta(days=30), importance=10,
    ))
    return store


@pytest.mark.asyncio
async def test_recency_ranking_uses_injected_clock(tmp_path):
    """The winner between two facts flips depending on which clock is injected.

    At `FROZEN`, the recency term (~0.99 for the 1-hour-old fact vs. ~0.0007
    for the 30-day-old one) outweighs the 0.3*importance gap (0.03 vs. 0.3),
    so the recent-but-unimportant fact wins: 0.525 vs. 0.300.
    At `FROZEN + 60 days`, both facts have aged past the point where recency
    contributes anything material (both terms collapse toward 0), so the
    fixed importance term decides and the old-but-important fact wins:
    0.030 vs. 0.300.
    Two retrievers built over the *same* store, differing only in the
    injected clock, must therefore disagree on the winner -- a store/retriever
    pair that ignored `clock=` would return the same order both times.
    """
    store = await _store_with_recency_importance_tradeoff(tmp_path)

    near_retr = FactRetriever(store, clock=lambda: FROZEN)
    near_out = await near_retr.fetch(
        FactQuery(subject="aria", type=FactType.EVENT, limit=2))
    assert near_out[0].content == "最近但不重要"

    far_retr = FactRetriever(store, clock=lambda: FROZEN + timedelta(days=60))
    far_out = await far_retr.fetch(
        FactQuery(subject="aria", type=FactType.EVENT, limit=2))
    assert far_out[0].content == "久远但重要"


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
    """The same on-disk fact is visible or filtered depending on which clock
    the store was built with.

    Relying on the gap between `FROZEN` and the real wall clock is fragile:
    that gap drifts by exactly one day for every day this test file sits
    unchanged, so an expiry boundary picked to land "before" or "after" the
    real clock today silently stops discriminating once enough time passes.
    Instead, pick one fact whose `expires_at` sits strictly between two
    clocks we control, and open the *same* db file with each clock: a store
    that ignored `clock=` (e.g. hardcoded to datetime.now) would show the
    fact identically through both, since neither now-vs-FROZEN relationship
    would matter.
    """
    db_path = tmp_path / "f.db"
    before_clock = FROZEN
    after_clock = FROZEN + timedelta(days=2)
    expires_at = FROZEN + timedelta(days=1)  # strictly between the two clocks

    writer = FactStore(db_path, clock=lambda: before_clock)
    await writer.init()
    await writer.write(Fact(
        subject="aria", content="临界事实", source=Source.LIFE_SIMULATED,
        type=FactType.EVENT, ts=FROZEN - timedelta(hours=1), importance=5,
        expires_at=expires_at,
    ))

    before_store = FactStore(db_path, clock=lambda: before_clock)
    await before_store.init()
    before_rows = await before_store.query(subject="aria", limit=10)
    assert [r.content for r in before_rows] == ["临界事实"]

    after_store = FactStore(db_path, clock=lambda: after_clock)
    await after_store.init()
    after_rows = await after_store.query(subject="aria", limit=10)
    assert [r.content for r in after_rows] == []
