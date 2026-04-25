"""Tests for persona models and loading."""

from pathlib import Path

import pytest

from persona_agent.persona.loader import load_persona
from persona_agent.persona.models import PersonaConfig, Identity, Trait
from persona_agent.persona.prompt_builder import PromptBuilder


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
