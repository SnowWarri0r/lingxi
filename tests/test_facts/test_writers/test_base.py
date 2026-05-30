from datetime import datetime

import pytest

from lingxi.facts.models import Fact, FactType, Source
from lingxi.facts.store import FactStore
from lingxi.facts.writers.base import WriterBase
from lingxi.facts.writers.life import LifeWriter


class FakeWriter(WriterBase):
    ALLOWED_SOURCE = Source.LIFE_SIMULATED
    SUBJECT_PATTERN = r"^aria$"


@pytest.fixture
async def store(tmp_path):
    s = FactStore(tmp_path / "f.db")
    await s.init()
    return s


@pytest.mark.asyncio
async def test_writer_accepts_matching_subject_and_source(store):
    w = FakeWriter(store)
    f = await w.write(
        subject="aria", content="x",
        type=FactType.EVENT, ts=datetime.now(),
    )
    assert f.source == Source.LIFE_SIMULATED
    assert f.subject == "aria"


@pytest.mark.asyncio
async def test_writer_rejects_wrong_subject(store):
    w = FakeWriter(store)
    with pytest.raises(ValueError, match="subject"):
        await w.write(
            subject="user:u1", content="x",
            type=FactType.EVENT, ts=datetime.now(),
        )


# ---------------------------------------------------------------------------
# Scorer + trigger integration tests (Task B.5)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_writer_calls_scorer_when_importance_is_none(tmp_path):
    class StubScorer:
        async def score_one(self, fact):
            return 9

    store = FactStore(tmp_path / "facts.db")
    await store.init()
    writer = LifeWriter(store, scorer=StubScorer())
    f = Fact(subject="aria", content="x", source=Source.LIFE_SIMULATED,
             type=FactType.EVENT, ts=datetime.now())
    await writer.write(f)
    rows = await store.query(subject="aria", limit=1)
    assert rows[0].importance == 9


@pytest.mark.asyncio
async def test_writer_skips_scorer_when_importance_preset(tmp_path):
    class FailingScorer:
        async def score_one(self, fact):
            raise AssertionError("scorer must not be called")

    store = FactStore(tmp_path / "facts.db")
    await store.init()
    writer = LifeWriter(store, scorer=FailingScorer())
    f = Fact(subject="aria", content="x", source=Source.LIFE_SIMULATED,
             type=FactType.EVENT, ts=datetime.now(), importance=4)
    await writer.write(f)
    rows = await store.query(subject="aria", limit=1)
    assert rows[0].importance == 4


@pytest.mark.asyncio
async def test_writer_calls_trigger_after_successful_write(tmp_path):
    observed: list[int] = []

    class StubTrigger:
        async def observe(self, n):
            observed.append(n)

    store = FactStore(tmp_path / "facts.db")
    await store.init()
    writer = LifeWriter(store, scorer=None, reflection_trigger=StubTrigger())
    f = Fact(subject="aria", content="x", source=Source.LIFE_SIMULATED,
             type=FactType.EVENT, ts=datetime.now(), importance=6)
    await writer.write(f)
    assert observed == [6]


@pytest.mark.asyncio
async def test_write_skip_scorer_bypasses_scorer(tmp_path):
    class FailingScorer:
        async def score_one(self, fact):
            raise AssertionError("scorer must not be called")

    store = FactStore(tmp_path / "facts.db")
    await store.init()
    writer = LifeWriter(store, scorer=FailingScorer())
    f = Fact(subject="aria", content="x", source=Source.LIFE_SIMULATED,
             type=FactType.EVENT, ts=datetime.now())  # importance=None
    await writer.write_skip_scorer(f)
    rows = await store.query(subject="aria", limit=1)
    assert rows[0].importance == 5  # neutral fallback


@pytest.mark.asyncio
async def test_write_skip_scorer_can_suppress_trigger(tmp_path):
    observed: list[int] = []

    class StubTrigger:
        async def observe(self, n):
            observed.append(n)

    store = FactStore(tmp_path / "facts.db")
    await store.init()
    writer = LifeWriter(store, scorer=None, reflection_trigger=StubTrigger())
    f = Fact(subject="aria", content="x", source=Source.LIFE_SIMULATED,
             type=FactType.EVENT, ts=datetime.now(), importance=7)
    await writer.write_skip_scorer(f, trigger_observation=False)
    assert observed == []  # trigger suppressed


@pytest.mark.asyncio
async def test_writer_no_scorer_leaves_importance_none(tmp_path):
    """When no scorer and importance is None, importance should remain None after write."""
    store = FactStore(tmp_path / "facts.db")
    await store.init()
    writer = LifeWriter(store)  # no scorer, no trigger
    f = Fact(subject="aria", content="x", source=Source.LIFE_SIMULATED,
             type=FactType.EVENT, ts=datetime.now())
    result = await writer.write(f)
    assert result.importance is None


@pytest.mark.asyncio
async def test_trigger_not_called_when_importance_is_none(tmp_path):
    """Trigger should NOT fire when importance remains None (no scorer assigned it)."""
    observed: list = []

    class StubTrigger:
        async def observe(self, n):
            observed.append(n)

    store = FactStore(tmp_path / "facts.db")
    await store.init()
    writer = LifeWriter(store, scorer=None, reflection_trigger=StubTrigger())
    f = Fact(subject="aria", content="x", source=Source.LIFE_SIMULATED,
             type=FactType.EVENT, ts=datetime.now())  # importance=None
    await writer.write(f)
    assert observed == []  # trigger must not fire for None importance
