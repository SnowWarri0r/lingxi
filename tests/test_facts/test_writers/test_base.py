from datetime import datetime

import pytest

from lingxi.facts.models import FactType, Source
from lingxi.facts.store import FactStore
from lingxi.facts.writers.base import WriterBase


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
