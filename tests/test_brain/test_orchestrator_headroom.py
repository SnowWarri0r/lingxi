"""The decisions with the most to remember were the ones that got cut off.

Two of 94 orchestrator calls on the live chat failed to parse, and both had
stopped at exactly the 700-token cap, partway through memory_writes. Each
fell back to the default decision, losing everything it had decided — on the
day he told her the most.
"""

from types import SimpleNamespace

import pytest

from lingxi.brain.orchestrator import StateDigest, decide


class _LLM:
    def __init__(self):
        self.kwargs = {}

    async def complete(self, **kw):
        self.kwargs = kw
        return SimpleNamespace(content='{"engage_level": 0.5, "register": "light", '
                                       '"fact_queries": [], "topic_anchor": "", '
                                       '"memory_writes": ["对方下个月要搬家"]}')


@pytest.mark.asyncio
async def test_a_long_decision_has_room_to_finish():
    llm = _LLM()
    await decide(llm, "下个月搬家", StateDigest(activity="", mood="", last_lived=[]), {})

    assert llm.kwargs["max_tokens"] >= 1200


@pytest.mark.asyncio
async def test_its_memory_writes_still_come_through():
    decision = await decide(_LLM(), "下个月搬家",
                            StateDigest(activity="", mood="", last_lived=[]), {})

    assert decision.memory_writes == ["对方下个月要搬家"]
