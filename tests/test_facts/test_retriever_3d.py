"""Tests for the 3D scoring fetch() introduced in B.6.

score = 0.5 * recency_decay(hours_old) + 0.3 * importance/10 + 0.2 * fts_rank
recency_decay(h) = exp(-0.01 * h)
"""

import pytest
from datetime import datetime, timedelta
from lingxi.facts.store import FactStore
from lingxi.facts.retriever import FactRetriever, FactQuery
from lingxi.facts.models import Fact, Source, FactType


@pytest.mark.asyncio
async def test_3d_ranking_prefers_high_importance_over_recent_trivia(tmp_path):
    """High-importance fact wins even when moderately older than a trivial one.

    With exp(-0.01 * h) recency, the formula gives:
      old_important  (48h, imp=9): 0.5*exp(-0.48) + 0.3*0.9 ≈ 0.579
      recent_trivial (24h, imp=2): 0.5*exp(-0.24) + 0.3*0.2 ≈ 0.453
    So old_important wins despite being older.
    """
    store = FactStore(tmp_path / "facts.db")
    await store.init()
    now = datetime.now()
    # Moderately old high-importance fact (48h ago, importance=9)
    old_important = Fact(
        subject="aria", content="跟用户讨论失眠", source=Source.USER_STATED,
        type=FactType.EVENT, ts=now - timedelta(hours=48), importance=9,
    )
    # More recent but trivial fact (24h ago, importance=2)
    recent_trivial = Fact(
        subject="aria", content="喝了杯水", source=Source.LIFE_SIMULATED,
        type=FactType.EVENT, ts=now - timedelta(hours=24), importance=2,
    )
    await store.write(old_important)
    await store.write(recent_trivial)
    r = FactRetriever(store)
    results = await r.fetch(FactQuery(subject="aria", limit=1))
    assert results[0].id == old_important.id


@pytest.mark.asyncio
async def test_3d_ranking_respects_recency_when_importance_equal(tmp_path):
    store = FactStore(tmp_path / "facts.db")
    await store.init()
    now = datetime.now()
    older = Fact(
        subject="aria", content="A", source=Source.LIFE_SIMULATED,
        type=FactType.EVENT, ts=now - timedelta(days=2), importance=5,
    )
    newer = Fact(
        subject="aria", content="B", source=Source.LIFE_SIMULATED,
        type=FactType.EVENT, ts=now - timedelta(hours=1), importance=5,
    )
    await store.write(older)
    await store.write(newer)
    r = FactRetriever(store)
    results = await r.fetch(FactQuery(subject="aria", limit=1))
    assert results[0].id == newer.id


@pytest.mark.asyncio
async def test_fetch_updates_last_accessed(tmp_path):
    store = FactStore(tmp_path / "facts.db")
    await store.init()
    now = datetime.now()
    f = Fact(
        subject="aria", content="x", source=Source.LIFE_SIMULATED,
        type=FactType.EVENT, ts=now, importance=5,
    )
    await store.write(f)
    r = FactRetriever(store)
    await r.fetch(FactQuery(subject="aria", limit=5))
    rows = await store.query(subject="aria", limit=1)
    assert rows[0].last_accessed is not None


@pytest.mark.asyncio
async def test_fetch_uses_fts_when_semantic_given(tmp_path):
    store = FactStore(tmp_path / "facts.db")
    await store.init()
    now = datetime.now()
    # Both equal recency + importance, fts should break the tie
    matching = Fact(
        subject="aria", content="光变曲线 数据分析",
        source=Source.LIFE_SIMULATED, type=FactType.EVENT,
        ts=now, importance=5,
    )
    other = Fact(
        subject="aria", content="去吃了顿火锅",
        source=Source.LIFE_SIMULATED, type=FactType.EVENT,
        ts=now, importance=5,
    )
    await store.write(matching)
    await store.write(other)
    r = FactRetriever(store)
    results = await r.fetch(FactQuery(subject="aria", semantic="光变曲线", limit=1))
    assert results[0].id == matching.id
