"""Tests for the conversation engine."""

import pytest

from lingxi.conversation.engine import ConversationEngine
from lingxi.memory.manager import MemoryManager


class TestConversationEngine:
    @pytest.mark.asyncio
    async def test_basic_chat(self, sample_persona, mock_llm, tmp_path):
        memory = MemoryManager(data_dir=str(tmp_path / "memory"))
        engine = ConversationEngine(
            persona=sample_persona,
            llm_provider=mock_llm,
            memory_manager=memory,
        )

        response = await engine.chat("Hello!")
        assert response == "This is a mock response."
        assert memory.short_term.turn_count == 2  # user + assistant

    @pytest.mark.asyncio
    async def test_directive_extraction(self, sample_persona, tmp_path):
        from tests.conftest import MockLLMProvider

        mock = MockLLMProvider(responses=[
            'Nice to meet you!\n===META===\n'
            '{"memory_writes": ["User\'s name is Alice"]}'
        ])
        memory = MemoryManager(data_dir=str(tmp_path / "memory"))
        engine = ConversationEngine(
            persona=sample_persona,
            llm_provider=mock,
            memory_manager=memory,
        )

        response = await engine.chat("I'm Alice")
        assert "===META===" not in response
        assert "memory_writes" not in response
        assert "Nice to meet you" in response

    @pytest.mark.asyncio
    async def test_mood_update(self, sample_persona, tmp_path):
        from tests.conftest import MockLLMProvider

        mock = MockLLMProvider(responses=[
            'That\'s wonderful!\n===META===\n'
            '{"mood": "excited"}'
        ])
        memory = MemoryManager(data_dir=str(tmp_path / "memory"))
        engine = ConversationEngine(
            persona=sample_persona,
            llm_provider=mock,
            memory_manager=memory,
        )

        await engine.chat("I found a new star!")
        assert engine._current_mood == "excited"
