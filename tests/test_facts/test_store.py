from datetime import datetime, timedelta
from pathlib import Path

import pytest

from lingxi.facts.models import Fact, FactType, Source
from lingxi.facts.store import FactStore


@pytest.fixture
async def store(tmp_path):
    s = FactStore(tmp_path / "facts.db")
    await s.init()
    return s


def make_fact(**overrides) -> Fact:
    defaults = dict(
        subject="aria",
        content="测试事件",
        source=Source.LIFE_SIMULATED,
        type=FactType.EVENT,
        ts=datetime.now(),
    )
    defaults.update(overrides)
    return Fact(**defaults)


@pytest.mark.asyncio
async def test_init_creates_schema(tmp_path):
    s = FactStore(tmp_path / "facts.db")
    await s.init()
    assert (tmp_path / "facts.db").exists()


@pytest.mark.asyncio
async def test_write_and_read_by_id(store):
    f = make_fact()
    await store.write(f)
    loaded = await store.get(f.id)
    assert loaded is not None
    assert loaded.content == "测试事件"
    assert loaded.subject == "aria"


@pytest.mark.asyncio
async def test_query_by_subject(store):
    await store.write(make_fact(subject="aria", content="aria 事件"))
    await store.write(make_fact(subject="user:u1", content="user 事件"))
    await store.write(make_fact(subject="npc:xiaomin", content="npc 事件"))

    arias = await store.query(subject="aria")
    assert len(arias) == 1
    assert arias[0].content == "aria 事件"


@pytest.mark.asyncio
async def test_query_by_subject_and_type(store):
    await store.write(make_fact(subject="aria", type=FactType.EVENT, content="事件"))
    await store.write(make_fact(subject="aria", type=FactType.PATTERN, content="规律"))

    events = await store.query(subject="aria", type=FactType.EVENT)
    assert len(events) == 1
    assert events[0].type == FactType.EVENT


@pytest.mark.asyncio
async def test_query_orders_by_ts_desc(store):
    base = datetime(2026, 5, 27, 10, 0)
    await store.write(make_fact(ts=base, content="老"))
    await store.write(make_fact(ts=base + timedelta(hours=1), content="新"))

    results = await store.query(subject="aria")
    assert results[0].content == "新"
    assert results[1].content == "老"


@pytest.mark.asyncio
async def test_query_respects_limit(store):
    for i in range(10):
        await store.write(make_fact(content=f"e{i}"))
    results = await store.query(subject="aria", limit=3)
    assert len(results) == 3


@pytest.mark.asyncio
async def test_supersedes_chain_excludes_old(store):
    """Fact superseded by another should not appear in default queries."""
    old = make_fact(content="老说法")
    await store.write(old)
    new = make_fact(content="新说法", supersedes=old.id)
    await store.write(new)

    results = await store.query(subject="aria", exclude_superseded=True)
    contents = [f.content for f in results]
    assert "新说法" in contents
    assert "老说法" not in contents


@pytest.mark.asyncio
async def test_supersedes_chain_includable(store):
    old = make_fact(content="老说法")
    await store.write(old)
    await store.write(make_fact(content="新说法", supersedes=old.id))

    results = await store.query(subject="aria", exclude_superseded=False)
    assert len(results) == 2


@pytest.mark.asyncio
async def test_expired_facts_filtered(store):
    past = make_fact(content="过期", expires_at=datetime.now() - timedelta(hours=1))
    await store.write(past)
    fresh = make_fact(content="新鲜", expires_at=datetime.now() + timedelta(hours=1))
    await store.write(fresh)

    results = await store.query(subject="aria")
    contents = [f.content for f in results]
    assert "新鲜" in contents
    assert "过期" not in contents


@pytest.mark.asyncio
async def test_fts_search(store):
    await store.write(make_fact(content="今天和外婆通电话", tags=["family", "call"]))
    await store.write(make_fact(content="改 paper 改到崩溃", tags=["work"]))

    family_hits = await store.search_fts("外婆")
    assert len(family_hits) == 1
    assert "外婆" in family_hits[0].content

    work_hits = await store.search_fts("paper")
    assert len(work_hits) == 1


@pytest.mark.asyncio
async def test_count_by_subject(store):
    await store.write(make_fact(subject="aria"))
    await store.write(make_fact(subject="aria"))
    await store.write(make_fact(subject="user:u1"))
    counts = await store.count_by_subject()
    assert counts["aria"] == 2
    assert counts["user:u1"] == 1


@pytest.mark.asyncio
async def test_store_persists_importance_and_last_accessed(tmp_path):
    from lingxi.facts.store import FactStore
    from lingxi.facts.models import Fact, Source, FactType
    from datetime import datetime

    store = FactStore(tmp_path / "facts.db")
    await store.init()
    now = datetime.now()
    f = Fact(
        subject="aria", content="x", source=Source.LIFE_SIMULATED,
        type=FactType.EVENT, ts=now, importance=8, last_accessed=now,
    )
    await store.write(f)
    rows = await store.query(subject="aria", limit=1)
    assert rows[0].importance == 8
    assert rows[0].last_accessed is not None
