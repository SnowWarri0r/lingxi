import pytest
from datetime import datetime, timedelta


@pytest.mark.asyncio
async def test_queue_and_pending(tmp_path):
    from lingxi.proactive.share_intent import ShareIntentStore
    s = ShareIntentStore(tmp_path)
    queued = await s.queue("fact1", "xiaomin", 0.7)
    assert queued is True
    pending = await s.pending()
    assert len(pending) == 1
    assert pending[0].fact_id == "fact1"
    assert pending[0].source_npc == "xiaomin"
    assert pending[0].significance == 0.7


@pytest.mark.asyncio
async def test_queue_respects_cooldown(tmp_path):
    from lingxi.proactive.share_intent import ShareIntentStore
    s = ShareIntentStore(tmp_path, cooldown_hours=24)
    assert await s.queue("fact1", "xiaomin", 0.7) is True
    # Same NPC, immediately retry — cooldown blocks
    assert await s.queue("fact2", "xiaomin", 0.8) is False
    # Different NPC — allowed
    assert await s.queue("fact3", "echo", 0.7) is True


@pytest.mark.asyncio
async def test_consume_removes_from_pending(tmp_path):
    from lingxi.proactive.share_intent import ShareIntentStore
    s = ShareIntentStore(tmp_path)
    await s.queue("fact1", "xiaomin", 0.7)
    await s.queue("fact2", "echo", 0.6)
    await s.consume("fact1")
    pending = await s.pending()
    assert len(pending) == 1
    assert pending[0].fact_id == "fact2"


@pytest.mark.asyncio
async def test_persists_across_instances(tmp_path):
    from lingxi.proactive.share_intent import ShareIntentStore
    s1 = ShareIntentStore(tmp_path)
    await s1.queue("fact1", "xiaomin", 0.7)
    s2 = ShareIntentStore(tmp_path)
    pending = await s2.pending()
    assert len(pending) == 1
