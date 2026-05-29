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
        # Mood is dynamic — surfaces in the focus reminder, not the
        # static system prompt (Phase 2 context refactor).
        builder = PromptBuilder(sample_persona)
        reminder = builder.build_turn_focus_reminder(current_mood="excited")
        assert reminder is not None
        assert "excited" in reminder

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

    def test_axes_section_renders_extreme_baseline(self):
        # Short-term axis_modulation was dropped with inner_life. The axes
        # section now surfaces only EXTREME baseline axes (≤3 or ≥8).
        from lingxi.persona.models import (
            DecisionAxes, DecisionAxis, Identity, PersonaConfig,
        )
        persona = PersonaConfig(
            name="T",
            identity=Identity(full_name="T"),
            decision_axes=DecisionAxes(
                conflict_style=DecisionAxis(score=2),  # extreme low → surfaces
            ),
        )
        builder = PromptBuilder(persona)
        prompt = builder.build_system_prompt()
        assert "行为指纹" in prompt
        assert "回避冲突" in prompt  # extreme-low conflict_style → low_label surfaces
        # No short-term modulation framing any more
        assert "此刻被推往" not in prompt

    def test_recent_proactive_messages_omitted_when_empty(self):
        from lingxi.persona.models import Identity, PersonaConfig
        persona = PersonaConfig(name="T", identity=Identity(full_name="T"))
        builder = PromptBuilder(persona)
        reminder = builder.build_turn_focus_reminder(
            recent_proactive_messages=None,
        )
        if reminder is not None:
            assert "你最近主动跟这位说过的话" not in reminder

class TestTraitBehaviorCue:
    def test_default_cue_empty(self):
        from lingxi.persona.models import Trait
        t = Trait(trait="好奇", intensity=0.9)
        assert t.behavior_cue == ""

    def test_high_intensity_cue_renders(self):
        from lingxi.persona.models import (
            Identity, PersonaConfig, PersonalityProfile, Trait,
        )
        persona = PersonaConfig(
            name="T",
            identity=Identity(full_name="T"),
            personality=PersonalityProfile(
                traits=[
                    Trait(
                        trait="好奇",
                        intensity=0.9,
                        behavior_cue="听到新概念会问一个具体细节",
                    ),
                ],
            ),
        )
        prompt = PromptBuilder(persona).build_system_prompt()
        # Both label and cue surface
        assert "好奇" in prompt
        assert "听到新概念会问一个具体细节" in prompt
        # Cue gets the "→" arrow prefix
        assert "**好奇** → 听到新概念会问一个具体细节" in prompt

    def test_high_intensity_no_cue_label_only(self):
        from lingxi.persona.models import (
            Identity, PersonaConfig, PersonalityProfile, Trait,
        )
        persona = PersonaConfig(
            name="T",
            identity=Identity(full_name="T"),
            personality=PersonalityProfile(
                traits=[Trait(trait="好奇", intensity=0.9, behavior_cue="")],
            ),
        )
        prompt = PromptBuilder(persona).build_system_prompt()
        # Label appears
        assert "好奇" in prompt
        # No "具体怎么显出来的" header when no traits have cues
        assert "具体怎么显出来的" not in prompt

    def test_mid_intensity_cue_not_rendered(self):
        # Cues are only for high-intensity traits (>0.7) — mid-traits
        # are rendered as a label-only secondary list
        from lingxi.persona.models import (
            Identity, PersonaConfig, PersonalityProfile, Trait,
        )
        persona = PersonaConfig(
            name="T",
            identity=Identity(full_name="T"),
            personality=PersonalityProfile(
                traits=[
                    Trait(
                        trait="温和",
                        intensity=0.5,
                        behavior_cue="说话语气柔",
                    ),
                ],
            ),
        )
        prompt = PromptBuilder(persona).build_system_prompt()
        assert "温和" in prompt
        assert "说话语气柔" not in prompt

    def test_aria_yaml_loads_behavior_cues(self):
        path = Path(__file__).parent.parent.parent / "config" / "personas" / "example_persona.yaml"
        if not path.exists():
            return
        persona = load_persona(path)
        cued = [t for t in persona.personality.traits if t.behavior_cue]
        # All 5 high-intensity traits should have cues configured
        assert len(cued) >= 4


class TestMessageHabits:
    def test_default_empty_not_rendered(self):
        from lingxi.persona.models import Identity, PersonaConfig
        persona = PersonaConfig(name="T", identity=Identity(full_name="T"))
        builder = PromptBuilder(persona)
        prompt = builder.build_system_prompt()
        # No populated fields → no section
        assert "打字的习惯" not in prompt

    def test_is_populated_returns_false_for_empty(self):
        from lingxi.persona.models import MessageHabits
        assert MessageHabits().is_populated() is False

    def test_is_populated_returns_true_when_any_field_set(self):
        from lingxi.persona.models import MessageHabits
        assert MessageHabits(punctuation_habit="x").is_populated() is True
        assert MessageHabits(coldness_markers=["x"]).is_populated() is True
        assert MessageHabits(avg_length="短").is_populated() is True

    def test_renders_when_populated(self):
        from lingxi.persona.models import (
            Identity, MessageHabits, PersonaConfig,
        )
        persona = PersonaConfig(
            name="T",
            identity=Identity(full_name="T"),
            message_habits=MessageHabits(
                avg_length="短",
                punctuation_habit="句号常省",
                coldness_markers=["单字回"],
                warmth_markers=["接具体细节"],
            ),
        )
        builder = PromptBuilder(persona)
        prompt = builder.build_system_prompt()
        assert "打字的习惯" in prompt
        assert "句号常省" in prompt
        assert "单字回" in prompt
        assert "接具体细节" in prompt

    def test_signature_phrases_quoted(self):
        from lingxi.persona.models import (
            Identity, MessageHabits, PersonaConfig,
        )
        persona = PersonaConfig(
            name="T",
            identity=Identity(full_name="T"),
            message_habits=MessageHabits(signature_phrases=["诶", "好像"]),
        )
        builder = PromptBuilder(persona)
        prompt = builder.build_system_prompt()
        assert '"诶"' in prompt
        assert '"好像"' in prompt

    def test_signature_phrases_capped_at_six(self):
        from lingxi.persona.models import (
            Identity, MessageHabits, PersonaConfig,
        )
        persona = PersonaConfig(
            name="T",
            identity=Identity(full_name="T"),
            message_habits=MessageHabits(
                signature_phrases=[f"p{i}" for i in range(20)],
            ),
        )
        builder = PromptBuilder(persona)
        prompt = builder.build_system_prompt()
        assert '"p0"' in prompt
        assert '"p5"' in prompt
        assert '"p6"' not in prompt

    def test_aria_yaml_loads_message_habits(self):
        path = Path(__file__).parent.parent.parent / "config" / "personas" / "example_persona.yaml"
        if not path.exists():
            return
        persona = load_persona(path)
        assert persona.message_habits.is_populated()
        assert persona.message_habits.coldness_markers
        assert persona.message_habits.warmth_markers
        # Aria's signature_phrases stays empty (we deliberately don't pre-fill)
        assert persona.message_habits.signature_phrases == []


class TestStyleConfig:
    def test_defaults(self):
        cfg = StyleConfig()
        assert cfg.speech_max_chars == 60
        # No default prefill — style emerges from L2 habits + emotion
        # behavioral_implication + decision axes (state-driven), not from
        # random context-blind opener injection.
        assert cfg.prefill_openers == []
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
