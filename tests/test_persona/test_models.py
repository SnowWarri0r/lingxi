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


class TestStyleConfig:
    def test_defaults(self):
        cfg = StyleConfig()
        assert cfg.speech_max_chars == 40
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
        assert persona.style.speech_max_chars == 40
        assert persona.sampling.temperature == 1.0
