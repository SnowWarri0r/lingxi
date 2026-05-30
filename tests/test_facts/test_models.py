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
