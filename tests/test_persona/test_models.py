"""Tests for persona models and loading."""

from pathlib import Path

import pytest

from lingxi.persona.loader import load_persona
from lingxi.persona.models import PersonaConfig, Identity, Trait, StyleConfig, SamplingConfig
from lingxi.persona.prompt_builder import PromptBuilder


class TestPersonaModels:
    def test_create_minimal_persona(self):
        persona = PersonaConfig(
            name="Test",
            identity=Identity(full_name="Test Person"),
        )
        assert persona.name == "Test"
        assert persona.identity.full_name == "Test Person"

    def test_trait_intensity_validation(self):
        t = Trait(trait="curious", intensity=0.8)
        assert t.intensity == 0.8

        with pytest.raises(Exception):
            Trait(trait="curious", intensity=1.5)

        with pytest.raises(Exception):
            Trait(trait="curious", intensity=-0.1)


class TestPersonaLoader:
    def test_load_example_persona(self):
        path = Path(__file__).parent.parent.parent / "config" / "personas" / "example_persona.yaml"
        if path.exists():
            persona = load_persona(path)
            assert persona.name == "Aria"
            assert persona.identity.full_name == "Aria Nightshade"

    def test_load_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            load_persona("/nonexistent/path.yaml")


class TestPromptBuilder:
    def test_build_basic_prompt(self, sample_persona):
        builder = PromptBuilder(sample_persona)
        prompt = builder.build_system_prompt()
        assert "Test Character" in prompt
        assert "curious" in prompt or "好奇" in prompt

    def test_prompt_includes_mood(self, sample_persona):
        builder = PromptBuilder(sample_persona)
        prompt = builder.build_system_prompt(current_mood="excited")
        assert "excited" in prompt

    def test_axes_section_omitted_when_all_neutral(self, sample_persona):
        # Defaults are all 5/10 — no axis is extreme, no modulation → skip
        builder = PromptBuilder(sample_persona)
        prompt = builder.build_system_prompt()
        assert "行为指纹" not in prompt

    def test_axes_section_renders_extreme_axis(self):
        from lingxi.persona.models import (
            DecisionAxes, DecisionAxis, Identity, PersonaConfig,
        )
        persona = PersonaConfig(
            name="T",
            identity=Identity(full_name="T"),
            decision_axes=DecisionAxes(
                conflict_style=DecisionAxis(score=2),
                time_horizon=DecisionAxis(score=8),
            ),
        )
        builder = PromptBuilder(persona)
        prompt = builder.build_system_prompt()
        assert "行为指纹" in prompt
        assert "回避冲突" in prompt
        assert "看长远" in prompt

    def test_axes_section_renders_modulation(self):
        from lingxi.inner_life.models import InnerState
        from lingxi.persona.models import (
            DecisionAxes, DecisionAxis, Identity, PersonaConfig,
        )
        persona = PersonaConfig(
            name="T",
            identity=Identity(full_name="T"),
            decision_axes=DecisionAxes(
                action_bias=DecisionAxis(score=5),  # neutral baseline
            ),
        )
        # action_bias is neutral but has active modulation → must surface
        inner = InnerState(axis_modulation={"action_bias": -2})
        builder = PromptBuilder(persona)
        prompt = builder.build_system_prompt(inner_state=inner)
        assert "行为指纹" in prompt
        assert "action_bias" not in prompt  # raw axis names should NOT appear
        assert "此刻被推往" in prompt


class TestStyleConfig:
    def test_defaults(self):
        cfg = StyleConfig()
        assert cfg.speech_max_chars == 60
        # Defaults are mostly empty so forced prefills don't become a tic.
        assert cfg.prefill_openers == ["", "", "", "", "嗯"]
        assert cfg.blacklist_phrases == []

    def test_custom_values(self):
        cfg = StyleConfig(
            speech_max_chars=80,
            prefill_openers=["哈"],
            blacklist_phrases=["据说"],
        )
        assert cfg.speech_max_chars == 80
        assert cfg.prefill_openers == ["哈"]

    def test_speech_max_chars_bounds(self):
        with pytest.raises(Exception):
            StyleConfig(speech_max_chars=0)
        with pytest.raises(Exception):
            StyleConfig(speech_max_chars=501)


class TestSamplingConfig:
    def test_defaults(self):
        cfg = SamplingConfig()
        assert cfg.temperature == 1.0
        assert cfg.top_p == 0.95

    def test_bounds(self):
        # temperature should be clamped to non-negative
        with pytest.raises(Exception):
            SamplingConfig(temperature=-0.1)

    def test_top_p_bounds(self):
        with pytest.raises(Exception):
            SamplingConfig(top_p=-0.01)
        with pytest.raises(Exception):
            SamplingConfig(top_p=1.1)


class TestPersonaConfigNewFields:
    def test_style_and_sampling_attached(self):
        persona = PersonaConfig(
            name="T",
            identity=Identity(full_name="T"),
        )
        assert persona.style.speech_max_chars == 60
        assert persona.sampling.temperature == 1.0


class TestDecisionAxes:
    def test_default_axes_all_neutral(self):
        from lingxi.persona.models import DecisionAxes
        axes = DecisionAxes()
        for name in DecisionAxes.AXIS_NAMES:
            assert axes.get(name).score == 5
            assert axes.get(name).confidence == "high"

    def test_axis_score_clamped_1_to_10(self):
        from lingxi.persona.models import DecisionAxis
        with pytest.raises(Exception):
            DecisionAxis(score=0)
        with pytest.raises(Exception):
            DecisionAxis(score=11)

    def test_effective_score_baseline_no_modulation(self):
        from lingxi.persona.models import DecisionAxes, DecisionAxis
        axes = DecisionAxes(conflict_style=DecisionAxis(score=2))
        assert axes.effective_score("conflict_style") == 2

    def test_effective_score_with_modulation(self):
        from lingxi.persona.models import DecisionAxes, DecisionAxis
        axes = DecisionAxes(action_bias=DecisionAxis(score=3))
        assert axes.effective_score("action_bias", {"action_bias": 2}) == 5
        assert axes.effective_score("action_bias", {"action_bias": -2}) == 1

    def test_effective_score_clamps_at_bounds(self):
        from lingxi.persona.models import DecisionAxes, DecisionAxis
        axes = DecisionAxes(time_horizon=DecisionAxis(score=9))
        # +3 would push to 12, must clamp at 10
        assert axes.effective_score("time_horizon", {"time_horizon": 3}) == 10
        axes2 = DecisionAxes(risk_appetite=DecisionAxis(score=2))
        assert axes2.effective_score("risk_appetite", {"risk_appetite": -5}) == 1

    def test_persona_loads_with_axes_from_yaml(self):
        path = Path(__file__).parent.parent.parent / "config" / "personas" / "example_persona.yaml"
        if not path.exists():
            return
        persona = load_persona(path)
        # Aria's defining axes from yaml
        assert persona.decision_axes.conflict_style.score == 2
        assert persona.decision_axes.time_horizon.score == 8
        assert persona.decision_axes.action_bias.score == 3

    def test_persona_without_axes_yaml_field_uses_defaults(self):
        # Backwards compat: yaml without decision_axes still loads
        persona = PersonaConfig(name="T", identity=Identity(full_name="T"))
        assert persona.decision_axes.risk_appetite.score == 5
