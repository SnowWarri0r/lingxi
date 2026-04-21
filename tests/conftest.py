"""Shared test fixtures."""

from __future__ import annotations

import pytest

from persona_agent.memory.manager import MemoryManager
from persona_agent.persona.models import (
    EmotionalBaseline,
    GoalDefinition,
    Identity,
    PersonaConfig,
    PersonalityProfile,
    SpeakingStyle,
    Trait,
)
from persona_agent.providers.base import CompletionResult, LLMProvider


class MockLLMProvider(LLMProvider):
    """Mock LLM provider for testing."""

    def __init__(self, responses: list[str] | None = None):
        self._responses = responses or ["This is a mock response."]
        self._call_count = 0

    async def complete(self, messages, system=None, max_tokens=4096, temperature=0.7, **kwargs):
        response = self._responses[self._call_count % len(self._responses)]
        self._call_count += 1
        return CompletionResult(content=response, model="mock")


@pytest.fixture
def sample_persona() -> PersonaConfig:
    return PersonaConfig(
        name="TestChar",
        version="1.0",
        identity=Identity(
            full_name="Test Character",
            age=25,
            occupation="Tester",
            background="A test character for unit tests.",
        ),
        personality=PersonalityProfile(
            traits=[
                Trait(trait="curious", intensity=0.8),
                Trait(trait="friendly", intensity=0.9),
            ],
            values=["honesty", "kindness"],
            fears=["bugs"],
        ),
        speaking_style=SpeakingStyle(
            tone="casual",
            vocabulary_level="normal",
            verbal_habits=["uses tech metaphors"],
            example_phrases=["That's like a recursive function!"],
        ),
        emotional_baseline=EmotionalBaseline(default_mood="happy"),
        goals=[
            GoalDefinition(description="Help with testing", priority=0.9),
        ],
    )


@pytest.fixture
def mock_llm() -> MockLLMProvider:
    return MockLLMProvider()


@pytest.fixture
def memory_manager(tmp_path) -> MemoryManager:
    return MemoryManager(data_dir=str(tmp_path / "memory"))
