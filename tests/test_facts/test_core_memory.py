import pytest
from datetime import datetime
from pathlib import Path

from lingxi.facts.models import Fact, FactType, Source
from lingxi.facts.store import FactStore


async def _store(tmp_path) -> FactStore:
    s = FactStore(Path(tmp_path) / "facts.db")
    await s.init()
    return s


@pytest.mark.asyncio
async def test_get_core_block_none_when_empty(tmp_path):
    s = await _store(tmp_path)
    assert await s.get_core_block("aria") is None


@pytest.mark.asyncio
async def test_get_core_block_returns_latest_unsuperseded(tmp_path):
    s = await _store(tmp_path)
    f1 = Fact(subject="aria", content="v1", source=Source.LLM_INFERRED,
              type=FactType.CORE, ts=datetime(2026, 5, 1, 9, 0))
    await s.write(f1)
    f2 = Fact(subject="aria", content="v2", source=Source.LLM_INFERRED,
              type=FactType.CORE, ts=datetime(2026, 5, 1, 10, 0), supersedes=f1.id)
    await s.write(f2)
    block = await s.get_core_block("aria")
    assert block is not None
    assert block.content == "v2"


@pytest.mark.asyncio
async def test_get_core_block_scoped_by_subject(tmp_path):
    s = await _store(tmp_path)
    await s.write(Fact(subject="aria", content="A", source=Source.LLM_INFERRED,
                       type=FactType.CORE, ts=datetime(2026, 5, 1, 9, 0)))
    await s.write(Fact(subject="user:feishu:x", content="U", source=Source.LLM_INFERRED,
                       type=FactType.CORE, ts=datetime(2026, 5, 1, 9, 0)))
    a = await s.get_core_block("aria")
    u = await s.get_core_block("user:feishu:x")
    assert a.content == "A"
    assert u.content == "U"


@pytest.mark.asyncio
async def test_retriever_get_core_block(tmp_path):
    from lingxi.facts.retriever import FactRetriever
    s = await _store(tmp_path)
    await s.write(Fact(subject="aria", content="hello", source=Source.LLM_INFERRED,
                       type=FactType.CORE, ts=datetime(2026, 5, 1, 9, 0)))
    r = FactRetriever(s)
    block = await r.get_core_block("aria")
    assert block is not None and block.content == "hello"
    assert await r.get_core_block("aria-missing") is None


@pytest.mark.asyncio
async def test_core_writer_allows_aria_and_user(tmp_path):
    from lingxi.facts.writers.core_memory import CoreMemoryWriter
    s = await _store(tmp_path)
    w = CoreMemoryWriter(s)
    await w.write(subject="aria", content="self note", type=FactType.CORE,
                  source=Source.LLM_INFERRED, ts=datetime(2026, 5, 1, 9, 0))
    await w.write(subject="user:feishu:x", content="about him", type=FactType.CORE,
                  source=Source.LLM_INFERRED, ts=datetime(2026, 5, 1, 9, 0))
    assert (await s.get_core_block("aria")).content == "self note"
    assert (await s.get_core_block("user:feishu:x")).content == "about him"


@pytest.mark.asyncio
async def test_core_writer_rejects_foreign_subject(tmp_path):
    from lingxi.facts.writers.core_memory import CoreMemoryWriter
    s = await _store(tmp_path)
    w = CoreMemoryWriter(s)
    with pytest.raises(ValueError):
        await w.write(subject="npc:bob", content="x", type=FactType.CORE,
                      source=Source.LLM_INFERRED, ts=datetime(2026, 5, 1, 9, 0))
