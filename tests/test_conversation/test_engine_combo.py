"""Tests for ConversationEngine single-call combo wiring.

Uses a FakeLLMProvider that records the arguments it received, so we
can assert prefill / sampler / style preamble are being passed through.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import AsyncIterator

import pytest

from lingxi.conversation.engine import ConversationEngine
from lingxi.memory.manager import MemoryManager
from lingxi.persona.models import (
    CompressionConfig,
    Identity,
    PersonaConfig,
    SamplingConfig,
    StyleConfig,
)
from lingxi.providers.base import CompletionResult, LLMProvider, StreamChunk


class FakeLLM(LLMProvider):
    def __init__(self):
        self.last_messages: list[dict] | None = None
        self.last_system: str | None = None
        self.last_kwargs: dict = {}

    async def complete(self, messages, system=None, max_tokens=4096,
                       temperature=0.7, top_p=None, prefill="", **kwargs):
        self.last_messages = list(messages)
        self.last_system = system
        self.last_kwargs = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "prefill": prefill,
        }
        # Echo a minimal response body
        return CompletionResult(content=f"{prefill}好的")

    async def complete_stream(self, messages, system=None, max_tokens=4096,
                              temperature=0.7, top_p=None, prefill="", **kwargs):
        result = await self.complete(
            messages, system, max_tokens, temperature, top_p, prefill, **kwargs
        )
        yield StreamChunk(content=result.content, is_final=True)

    async def embed(self, text: str):
        return [0.0] * 8


@pytest.fixture
def persona():
    return PersonaConfig(
        name="Test",
        identity=Identity(full_name="Test"),
        style=StyleConfig(
            speech_max_chars=30,
            prefill_openers=["嗯"],   # deterministic
            blacklist_phrases=["据说"],
        ),
        sampling=SamplingConfig(temperature=0.8, top_p=0.9),
    )


@pytest.fixture
def memory(tmp_path):
    return MemoryManager(data_dir=str(tmp_path), long_term_backend="json")


async def test_style_preamble_prepended_to_user_message(persona, memory):
    llm = FakeLLM()
    engine = ConversationEngine(persona=persona, llm_provider=llm, memory_manager=memory)
    await engine.chat("你今天吃啥", channel="cli", recipient_id="tester")

    # The last user message should start with the style preamble
    user_msgs = [m for m in llm.last_messages or [] if m["role"] == "user"]
    assert user_msgs, "no user message was sent"
    last_user = user_msgs[-1]
    content = last_user["content"]
    # Content may be a string or list of blocks
    if isinstance(content, list):
        # Find the text block
        text = next((b.get("text", "") for b in content if b.get("type") == "text"), "")
    else:
        text = content
    assert "[本轮" in text
    assert "≤30" in text
    assert "据说" in text  # persona blacklist merged
    assert "你今天吃啥" in text


async def test_sampler_forwarded(persona, memory):
    llm = FakeLLM()
    engine = ConversationEngine(persona=persona, llm_provider=llm, memory_manager=memory)
    await engine.chat("hi", channel="cli", recipient_id="tester")
    assert llm.last_kwargs["temperature"] == 0.8
    assert llm.last_kwargs["top_p"] == 0.9


async def test_prefill_forwarded(persona, memory):
    llm = FakeLLM()
    engine = ConversationEngine(persona=persona, llm_provider=llm, memory_manager=memory)
    output = await engine.chat_full("hi", channel="cli", recipient_id="tester")
    assert llm.last_kwargs["prefill"] == "嗯"


async def test_turn_id_populated(persona, memory):
    llm = FakeLLM()
    engine = ConversationEngine(persona=persona, llm_provider=llm, memory_manager=memory)
    output = await engine.chat_full("hi", channel="cli", recipient_id="tester")
    assert output.turn_id and len(output.turn_id) > 0


class _AItoneStreamLLM(LLMProvider):
    """FakeLLM whose compress_stream emits raw em-dashes + paragraph breaks.

    Used to verify the two-call streaming path does NOT leak unsanitized
    chunks before the final cleanup runs.
    """

    def __init__(self):
        self.last_messages: list[dict] | None = None
        self.last_system: str | None = None
        self.last_kwargs: dict = {}

    async def complete(self, messages, system=None, max_tokens=4096,
                       temperature=0.7, top_p=None, prefill="", **kwargs):
        self.last_messages = list(messages)
        self.last_system = system
        # Think-call return: well-formed "speech\n===META===\n{}" so the
        # parser produces a non-empty inner_thought for compress to use.
        return CompletionResult(content="想了想他在说什么\n===META===\n{\"inner\": \"想了想\"}")

    async def complete_stream(self, messages, system=None, max_tokens=4096,
                              temperature=0.7, top_p=None, prefill="", **kwargs):
        # Compress-call stream: simulate Haiku emitting AI-tone artifacts.
        for piece in ["嗯 我懂", " —— 我也", "有过\n\n那种感觉。"]:
            yield StreamChunk(content=piece, is_final=False)
        yield StreamChunk(content="", is_final=True)

    async def embed(self, text: str):
        return [0.0] * 8


async def test_two_call_stream_does_not_leak_uncleaned_chunks(memory):
    """Regression: two-call compression must not emit raw chunks containing
    em-dashes or multi-paragraph layout before final cleanup. The user
    should only see the cleaned final speech via the `done` event.
    """
    persona = PersonaConfig(
        name="Test",
        identity=Identity(full_name="Test"),
        style=StyleConfig(speech_max_chars=60, prefill_openers=[""]),
        sampling=SamplingConfig(temperature=1.0, top_p=0.95),
        compression=CompressionConfig(enabled=True),
    )
    llm = _AItoneStreamLLM()
    engine = ConversationEngine(persona=persona, llm_provider=llm, memory_manager=memory)

    events: list[tuple[str, str]] = []
    async for event in engine.chat_stream_events("hi", channel="cli", recipient_id="tester"):
        events.append((event.type, event.content))

    chunk_events = [c for t, c in events if t == "chunk"]
    assert chunk_events == [], (
        f"two-call path leaked uncleaned chunks to channel: {chunk_events!r}"
    )

    done_events = [c for t, c in events if t == "done"]
    assert len(done_events) == 1, "expected exactly one done event"
    final = done_events[0]
    assert "——" not in final, f"em-dash survived into done: {final!r}"
    assert " — " not in final, f"single em-dash survived into done: {final!r}"
    assert "\n\n" not in final, f"paragraph break survived into done: {final!r}"


async def test_stream_events_surfaces_turn_id(persona, memory):
    llm = FakeLLM()
    engine = ConversationEngine(persona=persona, llm_provider=llm, memory_manager=memory)

    events = []
    async for event in engine.chat_stream_events("hi", channel="cli", recipient_id="tester"):
        events.append((event.type, event.content))

    turn_id_events = [e for e in events if e[0] == "turn_id"]
    assert len(turn_id_events) == 1
    tid = turn_id_events[0][1]
    # Validate UUID format
    import uuid as _uuid
    _uuid.UUID(tid)  # raises ValueError if not a valid UUID

    # turn_id event must come before done event
    type_order = [e[0] for e in events]
    assert type_order.index("turn_id") < type_order.index("done")
