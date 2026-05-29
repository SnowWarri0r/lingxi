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


class _ScriptedLLM:
    """Emits a queued sequence of CompletionResults."""
    def __init__(self, results):
        self._results = list(results)
        self.calls = 0

    async def complete(self, **kwargs):
        self.calls += 1
        self._last_tool_choice = kwargs.get("tool_choice")
        return self._results.pop(0) if self._results else _final("forced")


def _toolcall(name, args, tid="t1"):
    from lingxi.providers.base import CompletionResult
    return CompletionResult(
        content="", finish_reason="tool_use",
        tool_calls=[{"id": tid, "name": name, "input": args}],
        raw_content_blocks=[{"type": "tool_use", "id": tid, "name": name, "input": args}],
    )


def _final(text):
    from lingxi.providers.base import CompletionResult
    return CompletionResult(content=text, finish_reason="end_turn")


@pytest.mark.asyncio
async def test_generate_with_tools_runs_tool_then_final(tmp_path):
    eng, store = await _engine(tmp_path)
    eng.llm = _ScriptedLLM([
        _toolcall("core_memory_append", {"block": "human", "content": "记一笔"}),
        _final("好的"),
    ])
    text = await eng._generate_with_tools(
        "SYS", [{"role": "user", "content": "hi"}], recipient_key="feishu:x")
    assert text == "好的"
    assert (await store.get_core_block("user:feishu:x")).content.endswith("记一笔")


@pytest.mark.asyncio
async def test_generate_with_tools_runaway_cap(tmp_path):
    eng, store = await _engine(tmp_path)
    # always tool_use → must stop at cap and force a text reply
    eng.llm = _ScriptedLLM([_toolcall("conversation_search", {"query": "x"}) for _ in range(20)])
    text = await eng._generate_with_tools(
        "SYS", [{"role": "user", "content": "hi"}], recipient_key="feishu:x")
    assert eng.llm.calls <= 6  # MAX_TOOL_ITERS(5) + 1 forced
    assert eng.llm._last_tool_choice == {"type": "none"}


@pytest.mark.asyncio
async def test_stream_events_runs_tool_loop(tmp_path):
    eng, store = await _engine(tmp_path)

    # Isolate the generation/emit path: stub prep so we don't invoke the
    # orchestrator (which would consume the scripted LLM queue).
    async def _fake_prep(ui, im, ch, rid):
        eng._current_recipient_key = "feishu:x"
        return "SYS", [{"role": "user", "content": ui}]
    eng._prepare_turn_v2 = _fake_prep

    eng.llm = _ScriptedLLM([
        _toolcall("core_memory_append", {"block": "human", "content": "流式记一笔"}),
        _final("流式回复"),
    ])
    events = [e async for e in eng.chat_stream_events("hi", channel="feishu", recipient_id="x")]
    done = [e for e in events if e.type == "done"]
    assert done and "流式回复" in done[-1].content
    block = await store.get_core_block("user:feishu:x")
    assert block is not None and block.content.endswith("流式记一笔")
