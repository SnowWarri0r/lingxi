import asyncio
import pytest


@pytest.mark.asyncio
async def test_trigger_fires_when_threshold_reached():
    from lingxi.facts.reflection_trigger import ReflectionTrigger

    fired = []
    class FakeReflector:
        async def reflect(self):
            fired.append(True)

    t = ReflectionTrigger(FakeReflector(), threshold=10, max_interval_seconds=999)
    await t.observe(3)
    await t.observe(3)
    await t.observe(4)  # accum=10, threshold hit
    await asyncio.sleep(0.05)  # let the create_task settle
    assert len(fired) == 1


@pytest.mark.asyncio
async def test_trigger_fires_after_max_interval():
    from lingxi.facts.reflection_trigger import ReflectionTrigger

    fired = []
    class FakeReflector:
        async def reflect(self):
            fired.append(True)

    t = ReflectionTrigger(FakeReflector(), threshold=999, max_interval_seconds=0.1)
    await t.observe(1)
    await asyncio.sleep(0.15)
    await t.observe(1)  # interval exceeded, fires
    await asyncio.sleep(0.05)
    assert len(fired) == 1


@pytest.mark.asyncio
async def test_trigger_resets_after_fire():
    from lingxi.facts.reflection_trigger import ReflectionTrigger

    fired = []
    class FakeReflector:
        async def reflect(self):
            fired.append(True)

    t = ReflectionTrigger(FakeReflector(), threshold=10, max_interval_seconds=999)
    for _ in range(4):
        await t.observe(3)  # 12 total — fires once
    await asyncio.sleep(0.05)
    for _ in range(2):
        await t.observe(3)  # 6 — under threshold post-reset
    await asyncio.sleep(0.05)
    assert len(fired) == 1


@pytest.mark.asyncio
async def test_trigger_does_not_propagate_reflector_failure():
    from lingxi.facts.reflection_trigger import ReflectionTrigger

    class BrokenReflector:
        async def reflect(self):
            raise RuntimeError("reflection failed")

    t = ReflectionTrigger(BrokenReflector(), threshold=5, max_interval_seconds=999)
    # Must not raise even though reflect throws
    await t.observe(10)
    await asyncio.sleep(0.05)
    # Trigger should still be in a usable state (no exception propagated to observe)
    await t.observe(1)  # also must not raise
