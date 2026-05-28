from datetime import datetime, timedelta

import pytest

from lingxi.facts.models import Fact, FactType, Source


class TestSource:
    def test_default_confidence_per_source(self):
        assert Source.USER_STATED.default_confidence == 1.0
        assert Source.LIFE_SIMULATED.default_confidence == 0.8
        assert Source.NPC_TICKER.default_confidence == 0.8
        assert Source.LLM_INFERRED.default_confidence == 0.5
        assert Source.WORLD_FETCH.default_confidence == 0.9
        assert Source.BIOGRAPHY.default_confidence == 1.0


class TestFact:
    def test_basic_construction(self):
        f = Fact(
            subject="aria",
            content="今早煮了泡面",
            source=Source.LIFE_SIMULATED,
            type=FactType.EVENT,
            ts=datetime(2026, 5, 27, 8, 0),
        )
        assert f.id  # auto-generated uuid
        assert f.confidence == 0.8  # default for LIFE_SIMULATED
        assert f.expires_at is None
        assert f.supersedes is None
        assert f.tags == []
        assert isinstance(f.written_at, datetime)

    def test_explicit_confidence_overrides_default(self):
        f = Fact(
            subject="user:oc_x",
            content="工作时间 11-21",
            source=Source.USER_STATED,
            type=FactType.PATTERN,
            ts=datetime.now(),
            confidence=0.95,
        )
        assert f.confidence == 0.95

    def test_subject_format_validation(self):
        # subject must be "aria" | "user:..." | "npc:..." | "world"
        with pytest.raises(ValueError):
            Fact(
                subject="random",
                content="x",
                source=Source.USER_STATED,
                type=FactType.EVENT,
                ts=datetime.now(),
            )

    def test_supersedes_link(self):
        old = Fact(
            subject="aria", content="老版本", source=Source.LLM_INFERRED,
            type=FactType.PATTERN, ts=datetime.now(),
        )
        new = Fact(
            subject="aria", content="新版本", source=Source.USER_STATED,
            type=FactType.PATTERN, ts=datetime.now(),
            supersedes=old.id,
        )
        assert new.supersedes == old.id

    def test_fact_has_importance_field_optional(self):
        from lingxi.facts.models import Fact, Source, FactType
        from datetime import datetime
        f = Fact(
            subject="aria",
            content="test",
            source=Source.LIFE_SIMULATED,
            type=FactType.EVENT,
            ts=datetime.now(),
        )
        assert f.importance is None
        assert f.last_accessed is None

    def test_fact_accepts_importance_and_last_accessed(self):
        from lingxi.facts.models import Fact, Source, FactType
        from datetime import datetime
        now = datetime.now()
        f = Fact(
            subject="aria",
            content="test",
            source=Source.LIFE_SIMULATED,
            type=FactType.EVENT,
            ts=now,
            importance=7,
            last_accessed=now,
        )
        assert f.importance == 7
        assert f.last_accessed == now


def test_plan_fact_type_exists():
    from lingxi.facts.models import FactType
    assert FactType.PLAN.value == "plan"


def test_plan_fact_type_round_trips_through_store(tmp_path):
    """Verify a PLAN fact persists and reads back as PLAN type."""
    import asyncio
    from datetime import datetime
    from lingxi.facts.store import FactStore
    from lingxi.facts.models import Fact, Source, FactType

    async def run():
        store = FactStore(tmp_path / "facts.db")
        await store.init()
        f = Fact(
            subject="aria",
            content="跑光变曲线第三组分析",
            source=Source.LIFE_SIMULATED,
            type=FactType.PLAN,
            ts=datetime.now(),
            tags=["time_window:09:00-12:00"],
        )
        await store.write(f)
        rows = await store.query(subject="aria", type=FactType.PLAN, limit=5)
        assert len(rows) == 1
        assert rows[0].type == FactType.PLAN

    asyncio.run(run())
