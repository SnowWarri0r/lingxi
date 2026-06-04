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
async def test_prepare_turn_v2_attaches_images_main_path(tmp_path, monkeypatch):
    # Regression: the main (fact_retriever-wired) path must route the current
    # turn through _build_user_message so image-only messages don't send empty
    # content (Anthropic 400). The fallback path alone doesn't catch this.
    eng, store = await _engine(tmp_path)
    from lingxi.brain import orchestrator as orch_mod
    from lingxi.brain import renderer as rend_mod
    from lingxi.brain.models import OrchestrationDecision

    async def _fake_decide(*a, **k):
        return OrchestrationDecision(register="warm", engage_level=0.5,
                                     fact_queries=[], skip=[], topic_anchor="")

    async def _fake_render(*a, **k):
        return ""

    monkeypatch.setattr(orch_mod, "decide", _fake_decide)
    monkeypatch.setattr(rend_mod, "render_dynamic_blocks", _fake_render)

    _sys, messages = await eng._prepare_turn_v2(
        "", [{"media_type": "image/jpeg", "data": "B64"}], "feishu", "x")
    last = messages[-1]
    assert last["role"] == "user"
    assert isinstance(last["content"], list)
    assert any(b["type"] == "image" for b in last["content"])
    assert any(b["type"] == "text" and b["text"].strip() for b in last["content"])


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


@pytest.mark.asyncio
async def test_search_stickers_returns_candidates(tmp_path):
    from lingxi.stickers.store import StickerStore
    from lingxi.stickers.models import Sticker
    eng, _ = await _engine(tmp_path)
    sstore = StickerStore(Path(tmp_path) / "stickers.db")
    await sstore.init()
    st = Sticker(file_path="/img/wuyu.png", content_hash="h1",
                 caption="无语翻白眼", emotion="无语", tags=["无语"],
                 when_to_use="对方说离谱的话")
    await sstore.add(st)
    eng.sticker_store = sstore

    out = await eng._dispatch_memory_tool("search_stickers", {"query": "无语"}, "feishu:x")
    assert "无语翻白眼" in out
    assert st.id in out                       # id shown so agent can send it
    assert eng._sticker_candidates["feishu:x"] == {st.id: "/img/wuyu.png"}
    # nothing sent yet
    assert eng._pending_stickers.get("feishu:x") is None


@pytest.mark.asyncio
async def test_send_sticker_by_id_sets_pending(tmp_path):
    from lingxi.stickers.store import StickerStore
    from lingxi.stickers.models import Sticker
    eng, _ = await _engine(tmp_path)
    sstore = StickerStore(Path(tmp_path) / "stickers.db")
    await sstore.init()
    st = Sticker(file_path="/img/a.png", content_hash="h1", caption="无语", tags=["无语"])
    await sstore.add(st)
    eng.sticker_store = sstore

    await eng._dispatch_memory_tool("search_stickers", {"query": "无语"}, "feishu:x")
    out = await eng._dispatch_memory_tool("send_sticker", {"sticker_id": st.id}, "feishu:x")
    assert "会发出去" in out
    assert eng._pending_stickers["feishu:x"] == "/img/a.png"


@pytest.mark.asyncio
async def test_send_sticker_rejects_unsearched_id(tmp_path):
    eng, _ = await _engine(tmp_path)
    # no search first → candidates empty
    out = await eng._dispatch_memory_tool("send_sticker", {"sticker_id": "bogus"}, "feishu:x")
    assert "不在候选里" in out
    assert eng._pending_stickers.get("feishu:x") is None


@pytest.mark.asyncio
async def test_send_sticker_once_per_turn(tmp_path):
    from lingxi.stickers.store import StickerStore
    from lingxi.stickers.models import Sticker
    eng, _ = await _engine(tmp_path)
    sstore = StickerStore(Path(tmp_path) / "stickers.db")
    await sstore.init()
    st = Sticker(file_path="/img/a.png", content_hash="h1", caption="无语", tags=["无语"])
    await sstore.add(st)
    eng.sticker_store = sstore
    await eng._dispatch_memory_tool("search_stickers", {"query": "无语"}, "feishu:x")
    first = await eng._dispatch_memory_tool("send_sticker", {"sticker_id": st.id}, "feishu:x")
    second = await eng._dispatch_memory_tool("send_sticker", {"sticker_id": st.id}, "feishu:x")
    assert "会发出去" in first
    assert "已经发过" in second


@pytest.mark.asyncio
async def test_search_stickers_no_store(tmp_path):
    eng, _ = await _engine(tmp_path)
    out = await eng._dispatch_memory_tool("search_stickers", {"query": "无语"}, "feishu:x")
    assert "未启用" in out


@pytest.mark.asyncio
async def test_stream_events_emits_sticker(tmp_path):
    from lingxi.stickers.store import StickerStore
    from lingxi.stickers.models import Sticker
    eng, _ = await _engine(tmp_path)
    sstore = StickerStore(Path(tmp_path) / "stickers.db")
    await sstore.init()
    st = Sticker(file_path="/img/wuyu.png", content_hash="h1",
                 caption="无语", emotion="无语", tags=["无语"])
    await sstore.add(st)
    eng.sticker_store = sstore

    async def _fake_prep(ui, im, ch, rid):
        eng._current_recipient_key = "feishu:x"
        eng._pending_stickers["feishu:x"] = None
        eng._sticker_candidates["feishu:x"] = {}
        return "SYS", [{"role": "user", "content": ui}]
    eng._prepare_turn_v2 = _fake_prep

    eng.llm = _ScriptedLLM([
        _toolcall("search_stickers", {"query": "无语"}, tid="t1"),
        _toolcall("send_sticker", {"sticker_id": st.id}, tid="t2"),
        _final("哈哈对啊"),
    ])
    events = [e async for e in eng.chat_stream_events(
        "hi", channel="feishu", recipient_id="x")]
    stickers = [e for e in events if e.type == "sticker"]
    assert len(stickers) == 1
    assert stickers[0].content == "/img/wuyu.png"
    assert eng._pending_stickers.get("feishu:x") is None


@pytest.mark.asyncio
async def test_prepare_turn_v2_injects_current_time(tmp_path, monkeypatch):
    # Regression: the reactive path must inject the time-awareness reminder so
    # Aria knows the current time. The pure-GA prompt refactor had dropped it.
    eng, store = await _engine(tmp_path)
    from lingxi.brain import orchestrator as orch_mod
    from lingxi.brain import renderer as rend_mod
    from lingxi.brain.models import OrchestrationDecision

    async def _fake_decide(*a, **k):
        return OrchestrationDecision(register="warm", engage_level=0.5,
                                     fact_queries=[], skip=[], topic_anchor="")

    async def _fake_render(*a, **k):
        return ""

    monkeypatch.setattr(orch_mod, "decide", _fake_decide)
    monkeypatch.setattr(rend_mod, "render_dynamic_blocks", _fake_render)

    _sys, messages = await eng._prepare_turn_v2("现在几点", None, "feishu", "x")
    last = messages[-1]
    assert last["role"] == "user"
    content = (last["content"] if isinstance(last["content"], str)
               else " ".join(b.get("text", "") for b in last["content"]))
    assert "当前真实时间" in content      # time-awareness section restored
    assert "现在几点" in content          # original user text preserved


@pytest.mark.asyncio
async def test_prepare_turn_v2_injects_fewshot_voice_anchors(tmp_path, monkeypatch):
    # The anti-翻译腔 lever: retrieved real-corpus lines must reach the system
    # prompt as a voice-cadence block. (Pure-GA refactor had cut this.)
    eng, store = await _engine(tmp_path)
    from lingxi.brain import orchestrator as orch_mod
    from lingxi.brain import renderer as rend_mod
    from lingxi.brain.models import OrchestrationDecision
    from lingxi.fewshot.models import FewShotSample

    async def _fake_decide(*a, **k):
        return OrchestrationDecision(register="warm", engage_level=0.5,
                                     fact_queries=[], skip=[], topic_anchor="")

    async def _fake_render(*a, **k):
        return ""

    monkeypatch.setattr(orch_mod, "decide", _fake_decide)
    monkeypatch.setattr(rend_mod, "render_dynamic_blocks", _fake_render)

    class _FakeRetriever:
        async def retrieve(self, query_text, recipient_key=None, k=4, threshold=0.5):
            return [FewShotSample(
                id="x", inner_thought="", corrected_speech="唉，孤独只能自己调解了",
                context_summary="孤独", tags=[], source="corpus")]

    eng.fewshot_retriever = _FakeRetriever()
    sys_prompt, _ = await eng._prepare_turn_v2("我好孤独", None, "feishu", "x")
    assert "你平时说话的语感" in sys_prompt
    assert "唉，孤独只能自己调解了" in sys_prompt
