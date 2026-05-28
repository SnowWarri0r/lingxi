import pytest
from datetime import datetime
from lingxi.facts.store import FactStore
from lingxi.facts.retriever import FactRetriever
from lingxi.facts.models import Fact, Source, FactType


@pytest.mark.asyncio
async def test_fetch_by_id_returns_fact(tmp_path):
    store = FactStore(tmp_path / "facts.db")
    await store.init()
    f = Fact(subject="aria", content="x", source=Source.LIFE_SIMULATED,
             type=FactType.EVENT, ts=datetime.now())
    await store.write(f)
    r = FactRetriever(store)
    found = await r.fetch_by_id(f.id)
    assert found is not None
    assert found.id == f.id


@pytest.mark.asyncio
async def test_fetch_by_id_returns_none_for_missing(tmp_path):
    store = FactStore(tmp_path / "facts.db")
    await store.init()
    r = FactRetriever(store)
    found = await r.fetch_by_id("nonexistent")
    assert found is None
