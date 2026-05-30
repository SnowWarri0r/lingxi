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


class TestBuildUserMessage:
    def test_plain_text_when_no_images(self):
        msg = ConversationEngine._build_user_message("hi", None)
        assert msg == {"role": "user", "content": "hi"}

    def test_image_only_never_empty_content(self):
        # image with empty caption → must NOT produce empty content (caused
        # Anthropic 400 "user messages must have non-empty content")
        msg = ConversationEngine._build_user_message(
            "", [{"media_type": "image/jpeg", "data": "BASE64DATA"}])
        assert msg["role"] == "user"
        assert isinstance(msg["content"], list)
        types = [b["type"] for b in msg["content"]]
        assert "image" in types and "text" in types
        text_block = next(b for b in msg["content"] if b["type"] == "text")
        assert text_block["text"].strip()  # non-empty fallback caption
        img_block = next(b for b in msg["content"] if b["type"] == "image")
        assert img_block["source"]["type"] == "base64"
        assert img_block["source"]["media_type"] == "image/jpeg"
        assert img_block["source"]["data"] == "BASE64DATA"

    def test_image_with_caption_keeps_text(self):
        msg = ConversationEngine._build_user_message(
            "这是什么", [{"media_type": "image/png", "data": "X"}])
        text_block = next(b for b in msg["content"] if b["type"] == "text")
        assert text_block["text"] == "这是什么"
