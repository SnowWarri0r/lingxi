"""The day she reflects on has to be the day she just lived, in order.

The store returns events newest-first, and the question prompt sliced
`recent[-50:]` — the tail of a descending list, so the OLDEST 50. Measured
across 33 logged reflection runs, 3 exceeded 50 events (86, 61, 53) and on
those the newest 36 / 11 / 3 moments were dropped: she reflected on the day
before last and never saw yesterday.

The same slice also handed the list over unreversed, so a real logged prompt
opened with 泡面端着站窗边吹风 (night) and closed with 走廊拐角 (that morning).
Read backwards, nothing causes anything.
"""

import json
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from lingxi.facts.models import Fact, FactType, Source
from lingxi.facts.reflector import Reflector
from lingxi.facts.retriever import FactRetriever
from lingxi.facts.store import FactStore
from lingxi.facts.writers.inference import InferenceWriter


class FakeLLM:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    async def complete(self, *, messages, system=None, **kw):
        self.calls.append(messages[0]["content"])
        return SimpleNamespace(
            content=self.responses.pop(0) if self.responses else "")


async def _questions_prompt(n_events, tmp_path):
    """Run a reflection over `n_events` events and return the question prompt.

    Event i happened i minutes after the start, so content order is time order.
    """
    store = FactStore(tmp_path / "facts.db")
    await store.init()
    start = datetime.now() - timedelta(hours=6)
    for i in range(n_events):
        await store.write(Fact(
            subject="aria", content=f"事件{i:03d}", source=Source.LIFE_SIMULATED,
            type=FactType.EVENT, ts=start + timedelta(minutes=i), importance=5))

    llm = FakeLLM(json.dumps(["q?"]), "一条想明白的事。")
    reflector = Reflector(llm, FactRetriever(store),
                          InferenceWriter(store, scorer=None))
    await reflector.reflect()
    return llm.calls[0]


def _events_in(prompt):
    body = prompt.split("我最近经历的事：", 1)[1]
    return [ln.strip().removeprefix("- ")
            for ln in body.strip().splitlines() if ln.strip()]


@pytest.mark.asyncio
async def test_a_long_day_reflects_on_its_newest_moments(tmp_path):
    events = _events_in(await _questions_prompt(80, tmp_path))

    assert len(events) == 50
    assert "事件079" in events, "the last thing that happened must be in there"
    assert "事件000" not in events, "the oldest moments are the ones to drop"


@pytest.mark.asyncio
async def test_the_day_is_told_forwards(tmp_path):
    events = _events_in(await _questions_prompt(80, tmp_path))

    assert events == sorted(events), "earlier moments come first"


@pytest.mark.asyncio
async def test_a_short_day_keeps_every_moment_and_still_reads_forwards(tmp_path):
    events = _events_in(await _questions_prompt(12, tmp_path))

    assert len(events) == 12
    assert events[0] == "事件000" and events[-1] == "事件011"
