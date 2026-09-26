"""A restart should not cost the next turn its recap.

thread_summary is what the orchestrator hands itself as 前情提要. It lived in
a dict on the engine, so each restart opened the next turn with
「（无——这是话题开始或重启）」 — 4 of the 12 logged turns that had a
previous turn, every one a return after hours or days away.
"""

from pathlib import Path

import pytest

from lingxi.conversation.engine import ConversationEngine
from lingxi.facts.retriever import FactRetriever
from lingxi.facts.store import FactStore
from lingxi.memory.manager import MemoryManager
from lingxi.persona.models import Identity, PersonaConfig


async def _engine(tmp_path):
    store = FactStore(Path(tmp_path) / "facts.db")
    await store.init()

    class _LLM:
        async def complete(self, **kw): ...

    return ConversationEngine(
        persona=PersonaConfig(name="A", identity=Identity(full_name="A")),
        llm_provider=_LLM(),
        memory_manager=MemoryManager(data_dir=str(Path(tmp_path) / "mem")),
        fact_retriever=FactRetriever(store),
    )


def _patch(monkeypatch, summary, seen):
    from lingxi.brain import orchestrator as orch_mod
    from lingxi.brain import renderer as rend_mod
    from lingxi.brain.models import OrchestrationDecision

    async def _decide(*a, **k):
        seen.append(k.get("prev_thread_summary", ""))
        return OrchestrationDecision(register="light", engage_level=0.5,
                                     fact_queries=[], topic_anchor="",
                                     thread_summary=summary)

    async def _render(*a, **k):
        return ""

    monkeypatch.setattr(orch_mod, "decide", _decide)
    monkeypatch.setattr(rend_mod, "render_dynamic_blocks", _render)


RECAP = "她前一晚说起想和队友一起站上台，话题停在那儿，他还没接"


@pytest.mark.asyncio
async def test_the_recap_reaches_the_first_turn_after_a_restart(tmp_path, monkeypatch):
    seen = []
    _patch(monkeypatch, RECAP, seen)
    before = await _engine(tmp_path)
    await before._prepare_turn_v2("晚点聊", None, "feishu", "x")

    after = await _engine(tmp_path)          # the restart
    await after._prepare_turn_v2("我回来了", None, "feishu", "x")

    assert seen[-1] == RECAP


@pytest.mark.asyncio
async def test_recaps_stay_per_recipient(tmp_path, monkeypatch):
    seen = []
    _patch(monkeypatch, RECAP, seen)
    eng = await _engine(tmp_path)
    await eng._prepare_turn_v2("晚点聊", None, "feishu", "x")

    fresh = await _engine(tmp_path)
    await fresh._prepare_turn_v2("你好", None, "feishu", "someone_else")

    assert seen[-1] == ""


@pytest.mark.asyncio
async def test_an_unreadable_file_starts_empty_instead_of_failing(tmp_path, monkeypatch):
    seen = []
    _patch(monkeypatch, RECAP, seen)
    (Path(tmp_path) / "mem").mkdir(parents=True, exist_ok=True)
    (Path(tmp_path) / "mem" / "thread_summaries.json").write_text("{not json", encoding="utf-8")

    eng = await _engine(tmp_path)
    await eng._prepare_turn_v2("在吗", None, "feishu", "x")

    assert seen[-1] == ""
