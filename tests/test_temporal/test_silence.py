"""Tests for silence-to-emotion mapping (#3 time weight)."""

from datetime import timedelta

from lingxi.persona.models import EmotionState
from lingxi.temporal.silence import compute_silence_emotion_deltas


class TestBuckets:
    def test_under_2h_returns_empty(self):
        assert compute_silence_emotion_deltas(timedelta(hours=1)) == {}
        assert compute_silence_emotion_deltas(timedelta(minutes=30)) == {}

    def test_2h_to_12h_期待(self):
        deltas = compute_silence_emotion_deltas(timedelta(hours=5))
        assert deltas == {"期待": 0.2}

    def test_12h_to_2d_想念_期待(self):
        deltas = compute_silence_emotion_deltas(timedelta(hours=18))
        assert "想念" in deltas
        assert "期待" in deltas

    def test_2d_to_7d_想念_失落(self):
        deltas = compute_silence_emotion_deltas(timedelta(days=4))
        assert deltas.get("想念") == 0.5
        assert deltas.get("失落") == 0.2

    def test_over_7d_heavy(self):
        deltas = compute_silence_emotion_deltas(timedelta(days=10))
        assert deltas.get("想念") == 0.6
        assert deltas.get("孤独") == 0.3
        assert deltas.get("失落") == 0.3


class TestApplyToEmotionState:
    def test_apply_actually_shifts_state(self):
        # Confirm the deltas are usable with EmotionState.apply_deltas
        state = EmotionState(dimensions={"平静": 0.5})
        deltas = compute_silence_emotion_deltas(timedelta(days=4))
        state.apply_deltas(deltas, volatility=0.5)
        assert state.dimensions.get("想念", 0) > 0
        assert state.dimensions.get("失落", 0) > 0
        # 平静 should remain (not in deltas) — apply_deltas is additive on
        # the named dims, doesn't reset existing
        assert "平静" in state.dimensions

    def test_short_silence_no_op(self):
        state = EmotionState(dimensions={"平静": 0.5})
        deltas = compute_silence_emotion_deltas(timedelta(minutes=10))
        state.apply_deltas(deltas)
        # No 想念 introduced
        assert "想念" not in state.dimensions

    def test_volatility_scales_blend(self):
        # Higher volatility → bigger jump toward target
        low = EmotionState()
        high = EmotionState()
        deltas = compute_silence_emotion_deltas(timedelta(days=4))
        low.apply_deltas(deltas, volatility=0.1)
        high.apply_deltas(deltas, volatility=0.9)
        # High-volatility should land closer to the target intensity
        assert high.dimensions["想念"] > low.dimensions["想念"]
