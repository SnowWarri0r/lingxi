import pytest
from pathlib import Path
from datetime import datetime

from lingxi.conversation.engine import ConversationEngine
from lingxi.memory.manager import MemoryManager
from lingxi.facts.store import FactStore
from lingxi.facts.retriever import FactRetriever
from lingxi.facts.writers.core_memory import CoreMemoryWriter
from lingxi.facts.writers.user_statement import UserStatementWriter
from lingxi.facts.writers.inference import InferenceWriter
from lingxi.facts.models import Fact, FactType, Source
from lingxi.persona.models import PersonaConfig, Identity


async def _engine(tmp_path):
    store = FactStore(Path(tmp_path) / "facts.db")
    await store.init()
    retr = FactRetriever(store)
    persona = PersonaConfig(name="Aria", identity=Identity(full_name="Aria"))

    class _LLM:  # not used by dispatch tests
        async def complete(self, **kw): ...

    eng = ConversationEngine(
        persona=persona, llm_provider=_LLM(),
        memory_manager=MemoryManager(data_dir=str(Path(tmp_path) / "mem")),
        fact_retriever=retr,
        inference_writer=InferenceWriter(store),
        user_statement_writer=UserStatementWriter(store),
        core_memory_writer=CoreMemoryWriter(store),
    )
    eng._current_recipient_key = "feishu:x"
    return eng, store


@pytest.mark.asyncio
async def test_dispatch_core_append_then_render(tmp_path):
    eng, store = await _engine(tmp_path)
    out = await eng._dispatch_memory_tool(
        "core_memory_append", {"block": "human", "content": "他在做天文"}, "feishu:x")
    assert "ok" in out.lower()
    block = await store.get_core_block("user:feishu:x")
    assert "他在做天文" in block.content


@pytest.mark.asyncio
async def test_dispatch_core_append_size_cap(tmp_path):
    eng, store = await _engine(tmp_path)
    big = "x" * 1600
    out = await eng._dispatch_memory_tool(
        "core_memory_append", {"block": "persona", "content": big}, "feishu:x")
    assert "full" in out.lower()
    assert await store.get_core_block("aria") is None


@pytest.mark.asyncio
async def test_dispatch_insert_scopes_subject(tmp_path):
    eng, store = await _engine(tmp_path)
    await eng._dispatch_memory_tool(
        "archival_memory_insert", {"content": "喜欢美式", "scope": "user"}, "feishu:x")
    facts = await store.query(subject="user:feishu:x")
    assert any("美式" in f.content for f in facts)


@pytest.mark.asyncio
async def test_dispatch_search_returns_facts(tmp_path):
    eng, store = await _engine(tmp_path)
    await store.write(Fact(subject="aria", content="今天看了仙女座",
                           source=Source.LLM_INFERRED, type=FactType.EVENT,
                           ts=datetime(2026, 5, 1, 9, 0), importance=6))
    out = await eng._dispatch_memory_tool(
        "archival_memory_search", {"query": "仙女座", "scope": "self"}, "feishu:x")
    assert "仙女座" in out
