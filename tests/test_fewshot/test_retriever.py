"""Tests for FewShotRetriever."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lingxi.fewshot.models import FewShotSample
from lingxi.fewshot.retriever import FewShotRetriever
from lingxi.fewshot.store import FewShotStore


def _embed(text: str, dim: int = 16) -> list[float]:
    rng = np.random.default_rng(abs(hash(text)) % (2**32))
    v = rng.normal(size=dim)
    v = v / np.linalg.norm(v)
    return v.tolist()


class FakeEmbedder:
    async def embed(self, text: str):
        return _embed(text)


async def _populate(store: FewShotStore):
    samples = [
        FewShotSample(id="cor1", inner_thought="想喝咖啡的心情",
                      corrected_speech="买一杯", context_summary="犯困",
                      source="user_correction"),
        FewShotSample(id="pos1", inner_thought="想喝咖啡的心情",
                      corrected_speech="走吧", context_summary="犯困",
                      source="positive"),
        FewShotSample(id="seed1", inner_thought="想喝咖啡的心情",
                      corrected_speech="嗯", context_summary="犯困",
                      source="seed"),
        FewShotSample(id="far", inner_thought="跟前面都不搭边的事",
                      corrected_speech="嗯嗯", context_summary="无关",
                      source="seed"),
    ]
    for s in samples:
        await store.add(s, embedding=_embed(s.inner_thought))


@pytest.fixture
async def retriever(tmp_path):
    store = FewShotStore(data_dir=tmp_path, embedding_dim=16)
    await store.init()
    await _populate(store)
    return FewShotRetriever(store=store, embedder=FakeEmbedder())


async def test_retrieve_returns_top_k(retriever):
    results = await retriever.retrieve(query_text="想喝咖啡的心情", k=3)
    assert len(results) <= 3
    ids = [r.id for r in results]
    # Similar ones should dominate the top
    assert "cor1" in ids or "pos1" in ids or "seed1" in ids


async def test_source_priority_user_correction_ranks_first(retriever):
    # With ties on similarity, user_correction should outrank positive which outranks seed
    results = await retriever.retrieve(query_text="想喝咖啡的心情", k=3)
    sources = [r.source for r in results]
    # user_correction appears before seed if both present
    if "user_correction" in sources and "seed" in sources:
        assert sources.index("user_correction") < sources.index("seed")


async def test_threshold_filter(retriever):
    # Setting threshold above maximum possible cosine similarity should drop everything
    results = await retriever.retrieve(query_text="想喝咖啡的心情", k=3, threshold=1.1)
    assert results == []
