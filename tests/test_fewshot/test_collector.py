"""Tests for AnnotationCollector."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lingxi.fewshot.collector import AnnotationCollector
from lingxi.fewshot.models import AnnotationTurn
from lingxi.fewshot.store import AnnotationStore, FewShotStore


class FakeEmbedder:
    async def embed(self, text: str):
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        v = rng.normal(size=16)
        v = v / np.linalg.norm(v)
        return v.tolist()


class StubSummarizer:
    async def summarize(self, turn: AnnotationTurn) -> tuple[str, list[str]]:
        return ("scenario summary", ["tag1", "tag2"])


@pytest.fixture
async def fixtures(tmp_path):
    ann = AnnotationStore(data_dir=tmp_path)
    pool = FewShotStore(data_dir=tmp_path, embedding_dim=16)
    await pool.init()
    collector = AnnotationCollector(
        annotation_store=ann,
        fewshot_store=pool,
        embedder=FakeEmbedder(),
        summarizer=StubSummarizer(),
    )
    # Seed a turn
    await ann.record(AnnotationTurn(
        turn_id="t1", recipient_key="cli:me",
        user_message="hi", inner_thought="想敷衍一下", speech="嗨",
    ))
    return collector, ann, pool


async def test_record_positive_creates_sample(fixtures):
    collector, ann, pool = fixtures
    await collector.record_positive("t1")
    assert await pool.count() == 1

    turn = await ann.get_turn("t1")
    assert turn.annotation == "positive"


async def test_record_negative_only_marks_turn(fixtures):
    collector, ann, pool = fixtures
    await collector.record_negative("t1")
    assert await pool.count() == 0
    turn = await ann.get_turn("t1")
    assert turn.annotation == "negative"


async def test_record_correction_creates_sample_with_original(fixtures):
    collector, ann, pool = fixtures
    await collector.record_correction("t1", correction="嗨嗨")
    assert await pool.count() == 1
    turn = await ann.get_turn("t1")
    assert turn.annotation == "negative"
    assert turn.correction == "嗨嗨"


async def test_correction_on_missing_turn_raises(fixtures):
    collector, *_ = fixtures
    with pytest.raises(KeyError):
        await collector.record_correction("ghost", correction="x")
