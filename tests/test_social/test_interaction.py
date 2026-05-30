import pytest
from datetime import datetime
from types import SimpleNamespace
from lingxi.facts.models import Fact, Source, FactType


class FakeLLM:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.systems = []
    async def complete(self, *, messages, system=None, **kw):
        self.systems.append(system or "")
        return SimpleNamespace(content=self.responses.pop(0))


@pytest.mark.asyncio
async def test_bidirectional_writes_both_sides(tmp_path):
    from lingxi.facts.store import FactStore
    from lingxi.facts.retriever import FactRetriever
    from lingxi.facts.writers.life import LifeWriter
    from lingxi.facts.writers.npc import NPCWriter
    from lingxi.social.interaction import bidirectional_interaction

    store = FactStore(tmp_path / "facts.db")
    await store.init()
    life_writer = LifeWriter(store, scorer=None)
    npc_writer = NPCWriter(store, scorer=None)
    llm = FakeLLM(
        "刚刚跟小敏聊她答辩的事，她声音有点紧。",
        "刚刚跟 Aria 说了下答辩的事，她让我别想太多。"
    )

    await bidirectional_interaction(
        llm=llm,
        retriever=FactRetriever(store),
        life_writer=life_writer,
        npc_writer=npc_writer,
        npc_id="xiaomin",
        npc_display="小敏",
        scenario="聊她下周的答辩",
    )

    aria_events = await store.query(subject="aria", type=FactType.EVENT, limit=5)
    npc_events = await store.query(subject="npc:xiaomin", type=FactType.EVENT, limit=5)
    assert len(aria_events) == 1
    assert len(npc_events) == 1
    # Both systems first-person
    assert any("你是 Aria" in s for s in llm.systems)
    assert any("你是 小敏" in s for s in llm.systems)


@pytest.mark.asyncio
async def test_bidirectional_handles_npc_side_failure(tmp_path):
    from lingxi.facts.store import FactStore
    from lingxi.facts.retriever import FactRetriever
    from lingxi.facts.writers.life import LifeWriter
    from lingxi.facts.writers.npc import NPCWriter
    from lingxi.social.interaction import bidirectional_interaction

    store = FactStore(tmp_path / "facts.db")
    await store.init()
    # First call succeeds (Aria side), second raises (NPC side)
    class HalfBrokenLLM:
        def __init__(self):
            self.call_count = 0
        async def complete(self, *, messages, system=None, **kw):
            self.call_count += 1
            if self.call_count == 1:
                return SimpleNamespace(content="aria view")
            raise RuntimeError("npc side failed")

    await bidirectional_interaction(
        llm=HalfBrokenLLM(),
        retriever=FactRetriever(store),
        life_writer=LifeWriter(store, scorer=None),
        npc_writer=NPCWriter(store, scorer=None),
        npc_id="xiaomin",
        npc_display="小敏",
        scenario="test",
    )

    # Aria side persisted, NPC side gracefully skipped
    aria_events = await store.query(subject="aria", type=FactType.EVENT, limit=5)
    npc_events = await store.query(subject="npc:xiaomin", type=FactType.EVENT, limit=5)
    assert len(aria_events) == 1
    assert len(npc_events) == 0
