"""Voice-anchor retrieval must not be handed an empty query.

An image-only message arrives with no text at all. That empty string went
straight to the embedding API, which rejects it (`400 MissingParameter`), so
voice anchors were silently off for exactly those turns.
"""

from pathlib import Path

import pytest

from lingxi.conversation.engine import ConversationEngine
from lingxi.facts.retriever import FactRetriever
from lingxi.facts.store import FactStore
from lingxi.memory.manager import MemoryManager
from lingxi.persona.models import Identity, PersonaConfig, ResponderConfig


class _Recorder:
    """A fewshot retriever that fails the way the embedding API does."""

    def __init__(self):
        self.queries: list[str] = []

    async def retrieve(self, query_text, recipient_key=None, k=4, threshold=0.5):
        self.queries.append(query_text)
        if not query_text.strip():
            raise RuntimeError(
                "Doubao embedding API error 400: MissingParameter")
        return []


@pytest.fixture
def stub_brain(monkeypatch):
    from lingxi.brain import orchestrator as orch_mod
    from lingxi.brain import renderer as rend_mod
    from lingxi.brain.models import OrchestrationDecision

    async def _decide(*a, **k):
        return OrchestrationDecision(
            register="light", engage_level=0.5, fact_queries=[], skip=[],
            topic_anchor="anchor")

    async def _render(*a, **k):
        return ""

    monkeypatch.setattr(orch_mod, "decide", _decide)
    monkeypatch.setattr(rend_mod, "render_dynamic_blocks", _render)


async def _engine(tmp_path, retriever):
    store = FactStore(Path(tmp_path) / "facts.db")
    await store.init()

    class _LLM:
        async def complete(self, **kw): ...

    return ConversationEngine(
        # doubao reads images itself, so an image-only turn keeps its empty
        # text — which is what used to reach the embedder.
        persona=PersonaConfig(
            name="Aria", identity=Identity(full_name="Aria"),
            responder=ResponderConfig(provider="doubao")),
        llm_provider=_LLM(),
        memory_manager=MemoryManager(data_dir=str(Path(tmp_path) / "mem")),
        fact_retriever=FactRetriever(store),
        fewshot_retriever=retriever,
    )


@pytest.mark.asyncio
async def test_an_empty_query_never_reaches_the_embedder(tmp_path, stub_brain):
    rec = _Recorder()
    eng = await _engine(tmp_path, rec)

    await eng._prepare_turn_v2(
        "", [{"data": "AAAA", "media_type": "image/png"}], "feishu", "oc_test")

    assert rec.queries == []


@pytest.mark.asyncio
async def test_a_turn_with_words_still_retrieves(tmp_path, stub_brain):
    rec = _Recorder()
    eng = await _engine(tmp_path, rec)

    await eng._prepare_turn_v2("今天好累啊", None, "feishu", "oc_test")

    assert rec.queries == ["今天好累啊"]
