from datetime import datetime, timedelta

import pytest

from lingxi.facts.models import Fact, FactType, Source
from lingxi.facts.retriever import FactQuery, FactRetriever
from lingxi.facts.store import FactStore


@pytest.fixture
async def populated(tmp_path):
    store = FactStore(tmp_path / "facts.db")
    await store.init()
    now = datetime.now()
    # aria lived events
    for i in range(3):
        await store.write(Fact(
            subject="aria", content=f"aria 事件 {i}",
            source=Source.LIFE_SIMULATED, type=FactType.EVENT,
            ts=now - timedelta(hours=i),
        ))
    # user patterns
    await store.write(Fact(
        subject="user:u1", content="工作 11-21",
        source=Source.USER_STATED, type=FactType.PATTERN,
        ts=now,
    ))
    # npc events
    await store.write(Fact(
        subject="npc:xiaomin", content="小敏改 paper",
        source=Source.NPC_TICKER, type=FactType.EVENT,
        ts=now,
    ))
    return FactRetriever(store)


@pytest.mark.asyncio
async def test_fetch_by_subject(populated):
    facts = await populated.fetch(FactQuery(subject="aria", limit=5))
    assert len(facts) == 3
    assert all(f.subject == "aria" for f in facts)


@pytest.mark.asyncio
async def test_fetch_by_subject_and_type(populated):
    facts = await populated.fetch(
        FactQuery(subject="aria", type=FactType.EVENT, limit=2)
    )
    assert len(facts) == 2


@pytest.mark.asyncio
async def test_fetch_with_semantic_query(populated):
    facts = await populated.fetch(
        FactQuery(subject="aria", semantic="事件", limit=5)
    )
    # FTS5 match on "事件"
    assert len(facts) >= 1
    assert all("事件" in f.content for f in facts)


@pytest.mark.asyncio
async def test_catalog_returns_counts_per_bucket(populated):
    """Catalog gives orchestrator a count summary without content,
    so it can decide which buckets to query without blowing context."""
    catalog = await populated.catalog()
    assert catalog["aria.event"] == 3
    assert catalog["user:u1.pattern"] == 1
    assert catalog["npc:xiaomin.event"] == 1
