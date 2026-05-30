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
