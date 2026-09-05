"""Voice anchors must not fail silently, and one bad row must not cost them all.

The anti-翻译腔 block had never rendered once — zero injections across every
log on disk — and nothing said why, because "nothing matched" and "retrieval
crashed" looked identical from outside. Against real user messages the best
match ran 0.17–0.41 while the gate stood at 0.50.
"""

from datetime import datetime

import pytest

from lingxi.fewshot.models import FewShotSample
from lingxi.fewshot.retriever import FewShotRetriever
from lingxi.fewshot.store import FewShotQueryResult


def _sample(speech: str, source: str = "seed") -> FewShotSample:
    return FewShotSample(
        id=speech, inner_thought="", corrected_speech=speech,
        context_summary="", tags=[], recipient_key=None, source=source,
        created_at=datetime(2026, 9, 3),
    )


class _Store:
    def __init__(self, results):
        self._results = results

    async def query(self, query_embedding, k, recipient_key=None):
        return self._results


class _Embedder:
    async def embed(self, text):
        return [1.0, 0.0]


def _retriever(pairs):
    return FewShotRetriever(
        _Store([FewShotQueryResult(sample=_sample(s), similarity=sim)
                for s, sim in pairs]),
        _Embedder(),
    )


@pytest.mark.asyncio
async def test_nothing_clearing_the_gate_is_reported(capsys):
    r = _retriever([("哈哈 挺你的。", 0.41), ("欸 你怎么说到点上了", 0.39)])

    assert await r.retrieve("今天上班好累啊", threshold=0.5) == []

    out = capsys.readouterr().out
    assert "0.41" in out and "0.5" in out, out


@pytest.mark.asyncio
async def test_a_clearing_match_is_returned_without_the_notice(capsys):
    r = _retriever([("哈哈 挺你的。", 0.55)])

    assert len(await r.retrieve("在吗", threshold=0.5)) == 1
    assert "no anchors" not in capsys.readouterr().out


@pytest.mark.asyncio
async def test_an_empty_pool_says_nothing(capsys):
    """No candidates is a different situation from candidates that all lost."""
    assert await _retriever([]).retrieve("在吗", threshold=0.5) == []
    assert "candidates" not in capsys.readouterr().out


@pytest.mark.asyncio
async def test_the_gate_reads_raw_similarity_not_the_boosted_score():
    """A boost must not smuggle a weak match past the threshold."""
    store = _Store([FewShotQueryResult(
        sample=_sample("哈哈 挺你的。", source="user_correction"),
        similarity=0.48)])

    assert await FewShotRetriever(store, _Embedder()).retrieve(
        "在吗", threshold=0.5) == []


@pytest.mark.asyncio
async def test_a_row_without_metadata_does_not_cost_the_whole_query():
    """Chroma returns None for a row stored without metadata.

    Reading it raised, and the caller catches everything and logs 「retrieve
    failed」 — so one bad row silently cost every anchor for that turn.
    """
    from lingxi.fewshot.store import FewShotStore

    store = object.__new__(FewShotStore)

    async def _init():
        return None

    store.init = _init
    store._collection = type("C", (), {"query": staticmethod(lambda **kw: {
        "ids": [["bad", "good"]],
        "metadatas": [[None, {"corrected_speech": "哈哈 挺你的。",
                              "source": "seed"}]],
        "distances": [[0.4, 0.5]],
    })})()

    out = await store.query(query_embedding=[1.0, 0.0], k=2)

    assert len(out) == 1
    assert out[0].sample.corrected_speech == "哈哈 挺你的。"
