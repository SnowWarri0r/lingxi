"""Tests for /pet/state endpoint — emotion family classification + state assembly."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from lingxi.persona.models import EmotionState
from lingxi.pet.state_endpoint import build_pet_state_app, classify_emotion_family


# --- classify_emotion_family unit tests --------------------------------------


class TestClassifyEmotionFamily:
    def test_none_returns_neutral(self):
        assert classify_emotion_family(None) == "NEUTRAL"

    def test_baseline_dims_below_threshold_neutral(self):
        # 平静 0.5 / 好奇 0.4 — neither is in any family, all family scores 0
        e = EmotionState(dimensions={"平静": 0.5, "好奇": 0.4, "温暖": 0.2})
        assert classify_emotion_family(e) == "NEUTRAL"

    def test_heavy_wins_when_dominant(self):
        e = EmotionState(dimensions={"悲伤": 0.7, "好奇": 0.2})
        assert classify_emotion_family(e) == "HEAVY"

    def test_flustered_wins_when_dominant(self):
        e = EmotionState(dimensions={"慌乱": 0.6})
        assert classify_emotion_family(e) == "FLUSTERED"

    def test_high_energy_when_dominant(self):
        e = EmotionState(dimensions={"兴奋": 0.8})
        assert classify_emotion_family(e) == "HIGH_ENERGY"

    def test_low_energy_when_dominant(self):
        e = EmotionState(dimensions={"疲惫": 0.7})
        assert classify_emotion_family(e) == "LOW_ENERGY"

    def test_provoked_when_dominant(self):
        # PROVOKED_DIMS uses 嗔/不爽/委屈/etc. (not 生气) per persona.models
        e = EmotionState(dimensions={"委屈": 0.6})
        assert classify_emotion_family(e) == "PROVOKED"

    def test_threshold_prevents_minor_flicker(self):
        # 悲伤 0.3 is right at the threshold floor — should not flip
        e = EmotionState(dimensions={"悲伤": 0.29, "平静": 0.5})
        assert classify_emotion_family(e) == "NEUTRAL"


# --- /pet/state endpoint integration tests -----------------------------------


def _make_mock_engine(
    *,
    dimensions: dict | None = None,
    activity_content: str | None = None,
    fact_retriever_present: bool = True,
    mood: str = "平静",
):
    """Build a minimal engine mock for /pet/state.

    Current activity is facts-driven now: the endpoint reads the latest
    aria.event fact via engine.fact_retriever (inner_life was retired).
    """
    engine = MagicMock()
    engine._emotion_state = EmotionState(dimensions=dimensions or {"平静": 0.5})
    engine._current_mood = mood

    if not fact_retriever_present:
        engine.fact_retriever = None
        return engine

    events = []
    if activity_content is not None:
        ev = MagicMock()
        ev.content = activity_content
        events = [ev]
    retriever = MagicMock()
    retriever.fetch = AsyncMock(return_value=events)
    engine.fact_retriever = retriever
    return engine


class TestPetStateEndpoint:
    def test_health_ok(self):
        engine = _make_mock_engine()
        client = TestClient(build_pet_state_app(engine))
        r = client.get("/pet/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}

    def test_state_returns_neutral_baseline(self):
        engine = _make_mock_engine()
        client = TestClient(build_pet_state_app(engine))
        r = client.get("/pet/state")
        assert r.status_code == 200
        body = r.json()
        assert body["engagement_mode"] == "full"
        assert body["emotion_family"] == "NEUTRAL"
        assert body["activity_kind"] is None
        assert "ts" in body
        assert body["mood_narrative"] == "平静"

    def test_state_heavy_emotion_drives_withdrawn_sprite(self):
        engine = _make_mock_engine(dimensions={"悲伤": 0.7})
        client = TestClient(build_pet_state_app(engine))
        body = client.get("/pet/state").json()
        # 悲伤 0.7 ≥ HEAVY threshold 0.5 → engagement_mode = withdrawn
        assert body["engagement_mode"] == "withdrawn"
        assert body["sprite"] == "withdrawn"  # mode wins over emotion family

    def test_state_flustered_emotion_drives_flustered_sprite(self):
        engine = _make_mock_engine(dimensions={"慌乱": 0.6})
        client = TestClient(build_pet_state_app(engine))
        body = client.get("/pet/state").json()
        assert body["engagement_mode"] == "flustered"
        assert body["sprite"] == "flustered"

    def test_state_activity_from_facts(self):
        engine = _make_mock_engine(activity_content="在改关于仙女座那段，卡在开头一句")
        client = TestClient(build_pet_state_app(engine))
        body = client.get("/pet/state").json()
        assert body["activity_kind"] is None  # no structured ActivityKind any more
        assert body["activity_name"] == "在改关于仙女座那段，卡在开头一句"[:40]

    def test_state_high_energy_emotion(self):
        engine = _make_mock_engine(dimensions={"兴奋": 0.7})
        client = TestClient(build_pet_state_app(engine))
        body = client.get("/pet/state").json()
        # 兴奋 < HEAVY/FLUSTERED thresholds → engagement_mode stays full
        assert body["engagement_mode"] == "full"
        assert body["emotion_family"] == "HIGH_ENERGY"

    def test_state_works_without_fact_retriever(self):
        engine = _make_mock_engine(fact_retriever_present=False)
        client = TestClient(build_pet_state_app(engine))
        body = client.get("/pet/state").json()
        assert body["activity_kind"] is None
        assert body["activity_name"] is None

    def test_state_handles_retriever_failure(self):
        engine = MagicMock()
        engine._emotion_state = EmotionState(dimensions={"平静": 0.5})
        engine._current_mood = "平静"
        retriever = MagicMock()
        retriever.fetch = AsyncMock(side_effect=RuntimeError("db gone"))
        engine.fact_retriever = retriever

        client = TestClient(build_pet_state_app(engine))
        r = client.get("/pet/state")
        assert r.status_code == 200
        assert r.json()["activity_name"] is None
