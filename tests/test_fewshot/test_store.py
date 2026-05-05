"""Tests for AnnotationStore (FewShotStore tested in Task 7)."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from lingxi.fewshot.models import AnnotationKind, AnnotationTurn, FewShotSample
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


import numpy as np

from lingxi.fewshot.store import FewShotStore


def _fake_embed(text: str, dim: int = 16) -> list[float]:
    rng = np.random.default_rng(abs(hash(text)) % (2**32))
    v = rng.normal(size=dim)
    v = v / np.linalg.norm(v)
    return v.tolist()


@pytest.fixture
async def fewshot_store(tmp_path):
    store = FewShotStore(data_dir=tmp_path, embedding_dim=16)
    await store.init()
    return store


async def test_add_and_query(fewshot_store):
    s = FewShotSample(
        id="s1", inner_thought="想喝咖啡", corrected_speech="去买一杯",
        context_summary="上午犯困", source="seed",
    )
    await fewshot_store.add(s, embedding=_fake_embed(s.inner_thought, 16))

    results = await fewshot_store.query(
        query_embedding=_fake_embed("想喝咖啡", 16),
        k=3,
    )
    assert len(results) == 1
    assert results[0].sample.id == "s1"


async def test_backup_jsonl_written(fewshot_store, tmp_path):
    s = FewShotSample(
        id="s2", inner_thought="x", corrected_speech="y",
        context_summary="z", source="seed",
    )
    await fewshot_store.add(s, embedding=_fake_embed(s.inner_thought, 16))
    backup = tmp_path / "fewshot" / "samples.jsonl"
    assert backup.exists()
    assert "s2" in backup.read_text(encoding="utf-8")


async def test_recipient_filter(fewshot_store):
    alice = FewShotSample(
        id="a", inner_thought="t", corrected_speech="a1",
        context_summary="c", recipient_key="cli:alice", source="positive",
    )
    bob = FewShotSample(
        id="b", inner_thought="t", corrected_speech="b1",
        context_summary="c", recipient_key="cli:bob", source="positive",
    )
    glob = FewShotSample(
        id="g", inner_thought="t", corrected_speech="g1",
        context_summary="c", recipient_key=None, source="seed",
    )
    for s in (alice, bob, glob):
        await fewshot_store.add(s, embedding=_fake_embed(s.id, 16))

    results = await fewshot_store.query(
        query_embedding=_fake_embed("t", 16),
        k=10,
        recipient_key="cli:alice",
    )
    ids = {r.sample.id for r in results}
    assert "a" in ids
    assert "g" in ids
    assert "b" not in ids


async def test_remove_returns_false_when_chroma_delete_fails(fewshot_store, monkeypatch):
    """Regression: if chroma delete fails, jsonl backup must NOT be rewritten.

    Previously remove() swallowed chroma exceptions then rewrote jsonl
    anyway, leaving the vector live but the audit copy gone — sample
    keeps influencing replies but is no longer visible for cleanup.
    """
    s = FewShotSample(
        id="will-fail", inner_thought="x", corrected_speech="y",
        context_summary="z", source="seed",
    )
    await fewshot_store.add(s, embedding=_fake_embed(s.id, 16))

    # Force chroma delete to raise
    def _broken_delete(**kwargs):
        raise RuntimeError("chroma down")
    monkeypatch.setattr(fewshot_store._collection, "delete", _broken_delete)

    ok = await fewshot_store.remove("will-fail")
    assert ok is False, "should return False when chroma delete fails"

    # Backup must still contain the sample (jsonl was NOT rewritten)
    backup = fewshot_store.backup_path
    assert backup.exists()
    assert "will-fail" in backup.read_text(encoding="utf-8")


async def test_remove_atomic_jsonl_uses_temp_rename(fewshot_store):
    """Regression: jsonl rewrite must go through .tmp + atomic rename so
    a crash during rewrite can't truncate the backup.
    """
    s = FewShotSample(
        id="rm-target", inner_thought="x", corrected_speech="y",
        context_summary="z", source="seed",
    )
    await fewshot_store.add(s, embedding=_fake_embed(s.id, 16))

    ok = await fewshot_store.remove("rm-target")
    assert ok is True

    # No leftover .tmp file after the operation
    backup = fewshot_store.backup_path
    tmp = backup.with_suffix(backup.suffix + ".tmp")
    assert not tmp.exists(), "leftover .tmp file means rewrite was non-atomic"
    # Sample is gone from backup
    assert "rm-target" not in backup.read_text(encoding="utf-8")
