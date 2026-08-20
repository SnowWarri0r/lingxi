from datetime import datetime, timedelta
from pathlib import Path

import pytest

from lingxi.conversation.engine import ConversationEngine
from lingxi.facts.retriever import FactRetriever
from lingxi.facts.store import FactStore
from lingxi.memory.manager import MemoryManager
from lingxi.persona.models import Identity, PersonaConfig


FROZEN = datetime(2026, 8, 19, 20, 20, 54)


async def _engine(tmp_path):
    store = FactStore(Path(tmp_path) / "facts.db", clock=lambda: FROZEN)
    await store.init()
    retr = FactRetriever(store, clock=lambda: FROZEN)
    persona = PersonaConfig(name="Aria", identity=Identity(full_name="Aria"))

    class _LLM:
        async def complete(self, **kw): ...

    return ConversationEngine(
        persona=persona, llm_provider=_LLM(),
        memory_manager=MemoryManager(data_dir=str(Path(tmp_path) / "mem")),
        fact_retriever=retr,
    )


@pytest.mark.asyncio
async def test_frozen_clock_makes_assembly_byte_identical(tmp_path, monkeypatch):
    """Same case, same clock, two assemblies — identical system and messages.

    This is the foundation of the whole harness: if two runs of one case do
    not produce the same prompt, every score downstream is meaningless.
    """
    from lingxi.brain import orchestrator as orch_mod
    from lingxi.brain import renderer as rend_mod
    from lingxi.brain.models import OrchestrationDecision

    async def _fake_decide(*a, **k):
        return OrchestrationDecision(
            register="light", engage_level=0.5, fact_queries=[], skip=[],
            topic_anchor="anchor", user_state="还在公司")

    async def _fake_render(*a, **k):
        return "【你此刻】固定内容"

    monkeypatch.setattr(orch_mod, "decide", _fake_decide)
    monkeypatch.setattr(rend_mod, "render_dynamic_blocks", _fake_render)

    eng = await _engine(tmp_path)
    eng.memory.add_turn("user", "想下班了")
    eng.memory.short_term.get_history()[-1].timestamp = FROZEN - timedelta(minutes=2)

    sys_a, msgs_a = await eng._prepare_turn_v2(
        "在干嘛", None, "feishu", "x", now=FROZEN)
    sys_b, msgs_b = await eng._prepare_turn_v2(
        "在干嘛", None, "feishu", "x", now=FROZEN)

    assert sys_a == sys_b
    assert msgs_a[-1]["content"] == msgs_b[-1]["content"]


@pytest.mark.asyncio
async def test_frozen_clock_reaches_the_time_block(tmp_path, monkeypatch):
    """The reminder states the frozen time, not the wall clock."""
    from lingxi.brain import orchestrator as orch_mod
    from lingxi.brain import renderer as rend_mod
    from lingxi.brain.models import OrchestrationDecision

    async def _fake_decide(*a, **k):
        return OrchestrationDecision(
            register="light", engage_level=0.5, fact_queries=[], skip=[],
            topic_anchor="")

    async def _fake_render(*a, **k):
        return ""

    monkeypatch.setattr(orch_mod, "decide", _fake_decide)
    monkeypatch.setattr(rend_mod, "render_dynamic_blocks", _fake_render)

    eng = await _engine(tmp_path)
    _sys, msgs = await eng._prepare_turn_v2(
        "在干嘛", None, "feishu", "x", now=FROZEN)
    text = msgs[-1]["content"]
    assert "2026-08-19 20:20" in text


@pytest.mark.asyncio
async def test_date_dividers_use_injected_clock(tmp_path):
    """A turn from the frozen day reads 今天, regardless of the wall clock."""
    from lingxi.conversation.context import ContextAssembler
    from lingxi.memory.manager import MemoryContext
    from lingxi.memory.short_term import ConversationTurn

    turns = [
        ConversationTurn(role="user", content="早",
                         timestamp=FROZEN - timedelta(hours=3)),
        ConversationTurn(role="assistant", content="早呀",
                         timestamp=FROZEN - timedelta(hours=3)),
    ]
    ctx = MemoryContext(short_term_turns=turns)
    out = ContextAssembler().assemble_messages(ctx, now=FROZEN)
    divider = next(m for m in out if m["content"].startswith("[——"))
    assert "今天" in divider["content"]
