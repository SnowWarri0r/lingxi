import json
import pytest
from datetime import datetime
from types import SimpleNamespace
from lingxi.facts.models import Fact, Source, FactType


class FakeLLM:
    def __init__(self, *responses: str):
        self.responses = list(responses)
        self.calls: list[str] = []
        self.system_calls: list[str] = []

    async def complete(self, *, messages, system=None, **kw):
        self.calls.append(messages[0]["content"])
        self.system_calls.append(system or "")
        return SimpleNamespace(content=self.responses.pop(0))


@pytest.mark.asyncio
async def test_reflector_writes_pattern_per_question(tmp_path):
    from lingxi.facts.store import FactStore
    from lingxi.facts.retriever import FactRetriever
    from lingxi.facts.writers.inference import InferenceWriter
    from lingxi.facts.reflector import Reflector

    store = FactStore(tmp_path / "facts.db")
    await store.init()
    now = datetime.now()
    for i in range(12):
        f = Fact(subject="aria", content=f"event {i}",
                 source=Source.LIFE_SIMULATED, type=FactType.EVENT,
                 ts=now, importance=5)
        await store.write(f)

    questions_json = json.dumps(["我最近为什么这么累？", "我对工作的态度变了吗？"])
    llm = FakeLLM(questions_json, "我累是因为连轴转。", "我对工作的热情确实少了。")
    retriever = FactRetriever(store)
    inference_writer = InferenceWriter(store, scorer=None)
    reflector = Reflector(llm, retriever, inference_writer)

    await reflector.reflect()

    patterns = await store.query(subject="aria", type=FactType.PATTERN, limit=10)
    assert len(patterns) == 2
    assert all("我" in p.content for p in patterns)
    assert all(p.importance == 8 for p in patterns)


@pytest.mark.asyncio
async def test_reflector_uses_first_person_aria_system(tmp_path):
    from lingxi.facts.store import FactStore
    from lingxi.facts.retriever import FactRetriever
    from lingxi.facts.writers.inference import InferenceWriter
    from lingxi.facts.reflector import Reflector

    store = FactStore(tmp_path / "facts.db")
    await store.init()
    for i in range(11):
        await store.write(Fact(
            subject="aria", content=f"e{i}", source=Source.LIFE_SIMULATED,
            type=FactType.EVENT, ts=datetime.now(), importance=5))

    llm = FakeLLM(json.dumps(["q?"]), "answer.")
    reflector = Reflector(llm, FactRetriever(store), InferenceWriter(store, scorer=None))
    await reflector.reflect()

    for sys_msg in llm.system_calls:
        assert "Aria" in sys_msg
        assert "她" not in sys_msg
    for prompt in llm.calls:
        assert "她" not in prompt


@pytest.mark.asyncio
async def test_reflector_skips_when_not_enough_facts(tmp_path):
    from lingxi.facts.store import FactStore
    from lingxi.facts.retriever import FactRetriever
    from lingxi.facts.writers.inference import InferenceWriter
    from lingxi.facts.reflector import Reflector

    store = FactStore(tmp_path / "facts.db")
    await store.init()
    # Only 3 facts — under default min_facts of 10
    for i in range(3):
        await store.write(Fact(
            subject="aria", content=f"e{i}", source=Source.LIFE_SIMULATED,
            type=FactType.EVENT, ts=datetime.now(), importance=5))

    llm = FakeLLM()  # no responses; if reflect calls LLM, IndexError
    reflector = Reflector(llm, FactRetriever(store), InferenceWriter(store, scorer=None))
    await reflector.reflect()  # must not raise

    patterns = await store.query(subject="aria", type=FactType.PATTERN, limit=10)
    assert len(patterns) == 0


@pytest.mark.asyncio
async def test_reflector_only_chews_events_newer_than_last_pattern(tmp_path):
    from datetime import timedelta
    from lingxi.facts.store import FactStore
    from lingxi.facts.retriever import FactRetriever
    from lingxi.facts.writers.inference import InferenceWriter
    from lingxi.facts.reflector import Reflector

    store = FactStore(tmp_path / "facts.db")
    await store.init()
    now = datetime.now()
    # 12 old events, all BEFORE an existing pattern (already reflected on)
    for i in range(12):
        await store.write(Fact(
            subject="aria", content=f"old event {i}",
            source=Source.LIFE_SIMULATED, type=FactType.EVENT,
            ts=now - timedelta(hours=5), importance=5))
    await store.write(Fact(
        subject="aria", content="已有的洞见",
        source=Source.LLM_INFERRED, type=FactType.PATTERN,
        ts=now - timedelta(hours=2), importance=8))
    # only 3 fresh events after the watermark — under min_facts
    for i in range(3):
        await store.write(Fact(
            subject="aria", content=f"new event {i}",
            source=Source.LIFE_SIMULATED, type=FactType.EVENT,
            ts=now, importance=5))

    llm = FakeLLM("should not be called")
    reflector = Reflector(llm, FactRetriever(store), InferenceWriter(store, scorer=None))
    await reflector.reflect()

    # old events sit below the watermark → not enough fresh material → skip
    assert llm.calls == []
    patterns = await store.query(subject="aria", type=FactType.PATTERN, limit=10)
    assert len(patterns) == 1
