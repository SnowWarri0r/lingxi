import asyncio
import json
import pytest
from datetime import datetime
from types import SimpleNamespace
from lingxi.facts.models import Fact, Source, FactType


class FakeLLM:
    """Records calls; returns a canned JSON array."""
    def __init__(self, canned: list[dict]):
        self.canned = canned
        self.calls: list[dict] = []
        self.system_calls: list[str] = []

    async def complete(self, *, messages, system=None, **kwargs):
        self.system_calls.append(system or "")
        self.calls.append({"messages": messages, "system": system})
        return SimpleNamespace(content=json.dumps(self.canned))


@pytest.mark.asyncio
async def test_scorer_batches_five_facts_into_one_call():
    from lingxi.facts.scorer import ImportanceScorer
    facts = [Fact(id=str(i), subject="aria", content=f"event {i}",
                  source=Source.LIFE_SIMULATED, type=FactType.EVENT,
                  ts=datetime.now()) for i in range(5)]
    llm = FakeLLM([{"id": f.id, "score": 5, "reason": "ok"} for f in facts])
    scorer = ImportanceScorer(llm, batch_size=5, flush_seconds=10)
    scores = await asyncio.gather(*[scorer.score_one(f) for f in facts])
    assert scores == [5, 5, 5, 5, 5]
    assert len(llm.calls) == 1  # one batched call


@pytest.mark.asyncio
async def test_scorer_falls_back_to_default_on_llm_failure():
    from lingxi.facts.scorer import ImportanceScorer
    class BrokenLLM:
        async def complete(self, **kw):
            raise RuntimeError("api down")
    f = Fact(subject="aria", content="x", source=Source.USER_STATED,
             type=FactType.EVENT, ts=datetime.now())
    scorer = ImportanceScorer(BrokenLLM(), batch_size=1, flush_seconds=0.1)
    score = await scorer.score_one(f)
    assert score == 7  # USER_STATED default


@pytest.mark.asyncio
async def test_scorer_uses_first_person_system_for_aria_subject():
    from lingxi.facts.scorer import ImportanceScorer
    f = Fact(subject="aria", content="x", source=Source.LIFE_SIMULATED,
             type=FactType.EVENT, ts=datetime.now())
    llm = FakeLLM([{"id": f.id, "score": 5, "reason": "ok"}])
    scorer = ImportanceScorer(llm, batch_size=1, flush_seconds=0.1)
    await scorer.score_one(f)
    assert "你是 Aria" in llm.system_calls[0]
    prompt_text = llm.calls[0]["messages"][0]["content"]
    assert "她" not in prompt_text


@pytest.mark.asyncio
async def test_scorer_buckets_aria_and_npc_separately():
    from lingxi.facts.scorer import ImportanceScorer
    aria_fact = Fact(subject="aria", content="x", source=Source.LIFE_SIMULATED,
                     type=FactType.EVENT, ts=datetime.now())
    npc_fact = Fact(subject="npc:xiaomin", content="y", source=Source.NPC_TICKER,
                    type=FactType.EVENT, ts=datetime.now())
    llm = FakeLLM([
        {"id": aria_fact.id, "score": 5, "reason": "a"},
        {"id": npc_fact.id, "score": 6, "reason": "b"},
    ])
    scorer = ImportanceScorer(llm, batch_size=2, flush_seconds=0.05)
    # batch_size=2 but different buckets → each flushes via timer, 2 calls
    results = await asyncio.gather(
        scorer.score_one(aria_fact),
        scorer.score_one(npc_fact),
    )
    assert len(llm.calls) == 2
    # Verify each system message matches its persona
    seen_systems = sorted(llm.system_calls)
    assert any("Aria" in s for s in seen_systems)
    assert any("xiaomin" in s for s in seen_systems)
