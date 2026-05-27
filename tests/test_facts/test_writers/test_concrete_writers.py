from datetime import datetime

import pytest

from lingxi.facts.models import FactType, Source
from lingxi.facts.store import FactStore
from lingxi.facts.writers.biography import BiographyLoader
from lingxi.facts.writers.inference import InferenceWriter
from lingxi.facts.writers.life import LifeWriter
from lingxi.facts.writers.npc import NPCWriter
from lingxi.facts.writers.user_statement import UserStatementWriter
from lingxi.facts.writers.world import WorldWriter


@pytest.fixture
async def store(tmp_path):
    s = FactStore(tmp_path / "f.db")
    await s.init()
    return s


@pytest.mark.parametrize("writer_cls,source,good,bad", [
    (LifeWriter,            Source.LIFE_SIMULATED, "aria",          "user:u1"),
    (NPCWriter,             Source.NPC_TICKER,     "npc:xiaomin",   "aria"),
    (UserStatementWriter,   Source.USER_STATED,    "user:oc_x",     "npc:xiaomin"),
    (InferenceWriter,       Source.LLM_INFERRED,   "user:oc_x",     "world"),
    (InferenceWriter,       Source.LLM_INFERRED,   "aria",          "world"),
    (WorldWriter,           Source.WORLD_FETCH,    "world",         "aria"),
    (BiographyLoader,       Source.BIOGRAPHY,      "aria",          "user:u1"),
])
@pytest.mark.asyncio
async def test_writer_accepts_allowed_and_rejects_disallowed(
    store, writer_cls, source, good, bad,
):
    w = writer_cls(store)
    # Allowed subject succeeds
    f = await w.write(
        subject=good, content="x",
        type=FactType.EVENT, ts=datetime.now(),
    )
    assert f.source == source
    # Disallowed subject raises
    with pytest.raises(ValueError):
        await w.write(
            subject=bad, content="y",
            type=FactType.EVENT, ts=datetime.now(),
        )
