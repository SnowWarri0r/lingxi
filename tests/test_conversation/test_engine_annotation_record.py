"""Tests that every chat turn records an AnnotationTurn."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lingxi.conversation.engine import ConversationEngine
from lingxi.fewshot.store import AnnotationStore
from lingxi.memory.manager import MemoryManager
from lingxi.persona.models import Identity, PersonaConfig
from lingxi.providers.base import CompletionResult, LLMProvider, StreamChunk


class FakeLLM(LLMProvider):
    async def complete(self, messages, system=None, max_tokens=4096,
                       temperature=0.7, top_p=None, prefill="", **kwargs):
        # Return speech + meta with an inner_thought so we can verify it's stored
        body = (
            f"{prefill}吃啥都行。\n"
            "===META===\n"
            '{"inner": "想随便应付一下，其实有点累"}'
        )
        return CompletionResult(content=body)

    async def complete_stream(self, messages, system=None, max_tokens=4096,
                              temperature=0.7, top_p=None, prefill="", **kwargs):
        result = await self.complete(
            messages, system, max_tokens, temperature, top_p, prefill, **kwargs
        )
        yield StreamChunk(content=result.content, is_final=True)

    async def embed(self, text):
        return [0.0] * 16


async def test_chat_records_annotation_turn(tmp_path: Path):
    ann_store = AnnotationStore(data_dir=tmp_path)
    persona = PersonaConfig(name="T", identity=Identity(full_name="T"))
    engine = ConversationEngine(
        persona=persona,
        llm_provider=FakeLLM(),
        memory_manager=MemoryManager(data_dir=str(tmp_path / "mem"), long_term_backend="json"),
        annotation_store=ann_store,
    )
    output = await engine.chat_full("你今天吃啥", channel="cli", recipient_id="tester")

    assert output.turn_id
    turn = await ann_store.get_turn(output.turn_id)
    assert turn is not None
    assert turn.user_message == "你今天吃啥"
    assert turn.speech  # non-empty
    assert turn.inner_thought == "想随便应付一下，其实有点累"
    assert turn.recipient_key == "cli:tester"
