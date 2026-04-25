"""Tests for AnnotationStore (FewShotStore tested in Task 7)."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from lingxi.fewshot.models import AnnotationTurn
from lingxi.fewshot.store import AnnotationStore


@pytest.fixture
def tmp_store(tmp_path: Path) -> AnnotationStore:
    return AnnotationStore(data_dir=tmp_path)


async def test_record_and_get(tmp_store):
    t = AnnotationTurn(
        turn_id="t1", recipient_key="cli:me",
        user_message="hi", inner_thought="greet", speech="嗨",
    )
    await tmp_store.record(t)
    loaded = await tmp_store.get_turn("t1")
    assert loaded is not None
    assert loaded.speech == "嗨"
    assert loaded.inner_thought == "greet"


async def test_get_missing_returns_none(tmp_store):
    assert await tmp_store.get_turn("nope") is None


async def test_update_annotation(tmp_store):
    t = AnnotationTurn(
        turn_id="t1", recipient_key="cli:me",
        user_message="hi", inner_thought="", speech="嗨",
    )
    await tmp_store.record(t)
    await tmp_store.update_annotation("t1", kind="negative", correction="哟")
    loaded = await tmp_store.get_turn("t1")
    assert loaded.annotation == "negative"
    assert loaded.correction == "哟"


async def test_update_missing_raises(tmp_store):
    with pytest.raises(KeyError):
        await tmp_store.update_annotation("ghost", kind="positive")


async def test_cleanup_unannotated_old_turns(tmp_store, tmp_path):
    # Create two turns, mark one as old-mtime
    t_new = AnnotationTurn(
        turn_id="new", recipient_key="cli:a",
        user_message="", inner_thought="", speech="",
    )
    t_old = AnnotationTurn(
        turn_id="old", recipient_key="cli:a",
        user_message="", inner_thought="", speech="",
    )
    await tmp_store.record(t_new)
    await tmp_store.record(t_old)

    old_path = tmp_path / "turns" / "old.json"
    import os
    old_time = (datetime.now() - timedelta(days=45)).timestamp()
    os.utime(old_path, (old_time, old_time))

    deleted = await tmp_store.cleanup(max_age_unannotated_days=30)
    assert deleted == 1
    assert await tmp_store.get_turn("new") is not None
    assert await tmp_store.get_turn("old") is None
