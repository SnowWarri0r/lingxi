"""Tests for LifeSimulator._update_axis_modulation — derive axis deltas from state."""

from lingxi.inner_life.models import InnerState
from lingxi.inner_life.simulator import LifeSimulator


def _state(**overrides) -> InnerState:
    s = InnerState()
    for k, v in overrides.items():
        setattr(s, k, v)
    return s


class TestAxisModulationFromEnergy:
    def test_low_energy_drops_action_bias_and_conflict_style(self):
        s = _state(energy=0.2)
        LifeSimulator._update_axis_modulation(s)
        assert s.axis_modulation.get("action_bias") == -1
        assert s.axis_modulation.get("conflict_style") == -1

    def test_high_energy_raises_action_bias(self):
        s = _state(energy=0.9)
        LifeSimulator._update_axis_modulation(s)
        assert s.axis_modulation.get("action_bias") == 1

    def test_neutral_energy_no_modulation(self):
        s = _state(energy=0.6)
        LifeSimulator._update_axis_modulation(s)
        assert s.axis_modulation == {}


class TestAxisModulationFromSocial:
    def test_low_social_need_lowers_emotion_weight(self):
        s = _state(energy=0.6, social_need=0.1)
        LifeSimulator._update_axis_modulation(s)
        assert s.axis_modulation.get("emotion_weight") == -1
        assert s.axis_modulation.get("action_bias") == -1

    def test_high_social_need_raises_emotion_weight(self):
        s = _state(energy=0.6, social_need=0.9)
        LifeSimulator._update_axis_modulation(s)
        assert s.axis_modulation.get("emotion_weight") == 1


class TestAxisModulationCombined:
    def test_low_energy_and_low_social_compound_action_bias(self):
        # Both nudge action_bias down → -2 total (capped at -2)
        s = _state(energy=0.2, social_need=0.1)
        LifeSimulator._update_axis_modulation(s)
        assert s.axis_modulation.get("action_bias") == -2

    def test_modulation_capped_at_minus_two(self):
        # Even with extreme stacking, no axis exceeds ±2
        s = _state(energy=0.1, social_need=0.05, sleep_quality=0.2)
        LifeSimulator._update_axis_modulation(s)
        for delta in s.axis_modulation.values():
            assert -2 <= delta <= 2

    def test_replaces_not_accumulates(self):
        # Calling twice with same state gives same modulation
        s = _state(energy=0.2)
        LifeSimulator._update_axis_modulation(s)
        first = dict(s.axis_modulation)
        LifeSimulator._update_axis_modulation(s)
        assert s.axis_modulation == first


class TestAxisModulationFromSleep:
    def test_bad_sleep_lowers_novelty_and_risk(self):
        s = _state(energy=0.6, sleep_quality=0.3)
        LifeSimulator._update_axis_modulation(s)
        assert s.axis_modulation.get("novelty_seeking") == -1
        assert s.axis_modulation.get("risk_appetite") == -1
