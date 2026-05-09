"""Schema tests for relational memory."""

from datetime import datetime

from lingxi.relational.models import (
    DailyPattern,
    FightPattern,
    InsideJoke,
    RelationalMemory,
    SharedPlace,
    SweetMoment,
)


class TestEmptyMemory:
    def test_blank_memory_is_empty(self):
        m = RelationalMemory(recipient_key="x")
        assert m.is_empty()

    def test_with_inside_joke_not_empty(self):
        m = RelationalMemory(
            recipient_key="x",
            inside_jokes=[InsideJoke(phrase="x", origin="y")],
        )
        assert not m.is_empty()

    def test_with_pet_name_not_empty(self):
        m = RelationalMemory(recipient_key="x", pet_names=["笨蛋"])
        assert not m.is_empty()

    def test_with_summary_not_empty(self):
        m = RelationalMemory(recipient_key="x", relationship_summary="a paragraph")
        assert not m.is_empty()


class TestSubmodelDefaults:
    def test_inside_joke_has_default_timestamp(self):
        j = InsideJoke(phrase="x", origin="y")
        assert isinstance(j.last_used_at, datetime)
        assert j.use_count == 1

    def test_sweet_moment_default_weight_medium(self):
        m = SweetMoment(timestamp=datetime.now(), content="x")
        assert m.weight == "medium"

    def test_daily_pattern_default_confidence_medium(self):
        d = DailyPattern(pattern="x")
        assert d.confidence == "medium"

    def test_fight_pattern_optional_last_occurred(self):
        f = FightPattern(trigger="t", her_pattern="h", typical_repair="r")
        assert f.last_occurred_at is None

    def test_shared_place_default_referenced(self):
        p = SharedPlace(name="x", significance="y")
        assert isinstance(p.last_referenced_at, datetime)
