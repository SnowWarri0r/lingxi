"""Tests for InnerLifeStore.update_state — atomic read-modify-write.

Codex P2: proactive's _mark_event_shared previously did load_state →
mutate → save_state, with no lock between load and save. A concurrent
sim tick or reactive turn save_state in that window would clobber the
flag clear, or the proactive save would clobber the concurrent writes.

update_state re-reads under the lock so the mutator always operates on
the latest disk snapshot.
"""

from datetime import datetime

import pytest

from lingxi.inner_life.models import InnerState, LifeEvent
from lingxi.inner_life.store import InnerLifeStore


@pytest.fixture
def store(tmp_path):
    return InnerLifeStore(tmp_path)


@pytest.mark.asyncio
async def test_update_state_sees_latest_disk_snapshot(store):
    # First writer: state with flag=True, no last_chat_at
    s1 = InnerState(recent_events=[LifeEvent(content="x", wants_to_share=True)])
    await store.save_state(s1)

    # Simulated concurrent writer: persists state with last_chat_at set
    # AFTER the first save (mimics sim tick / reactive turn updating)
    later = datetime(2026, 5, 8, 12, 0)
    s2 = InnerState(
        recent_events=[LifeEvent(content="x", wants_to_share=True)],
        last_chat_at=later,
    )
    await store.save_state(s2)

    # update_state mutator clears the flag — must see s2's last_chat_at
    seen: list = []

    def mutator(state):
        seen.append(state.last_chat_at)
        for ev in state.recent_events:
            ev.wants_to_share = False

    await store.update_state(mutator)

    assert seen and seen[0] == later, "mutator must see latest state, not stale"

    final = await store.load_state()
    assert all(not ev.wants_to_share for ev in final.recent_events)
    assert final.last_chat_at == later, "concurrent writes must survive update_state"


@pytest.mark.asyncio
async def test_update_state_with_no_existing_file(store):
    # Should treat missing file as a fresh InnerState, not crash
    def mutator(state):
        state.energy = 0.42

    result = await store.update_state(mutator)
    assert result.energy == 0.42

    loaded = await store.load_state()
    assert loaded.energy == 0.42


@pytest.mark.asyncio
async def test_update_state_returns_post_mutation_state(store):
    initial = InnerState(energy=0.5)
    await store.save_state(initial)

    def mutator(state):
        state.energy = 0.9

    result = await store.update_state(mutator)
    assert result.energy == 0.9


@pytest.mark.asyncio
async def test_update_state_serializes_concurrent_callers(store):
    # Two update_state calls in flight: both must commit, neither lost.
    # Lock guarantees serialization, so final increment count matches calls.
    initial = InnerState(significant_events_today=0)
    await store.save_state(initial)

    def increment(state):
        state.significant_events_today += 1

    import asyncio
    await asyncio.gather(*[store.update_state(increment) for _ in range(10)])

    final = await store.load_state()
    assert final.significant_events_today == 10
