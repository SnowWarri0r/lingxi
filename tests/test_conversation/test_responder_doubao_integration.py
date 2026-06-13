"""Responder=doubao: the chat reply is one coherent pass on the live doubao
model, routed through the real engine path. Image turns ride doubao too (the
blocks get converted to OpenAI format). Live tests skip without ARK_API_KEY."""

import os

import pytest

from lingxi.conversation.engine import ConversationEngine
from lingxi.memory.manager import MemoryManager


_needs_ark = pytest.mark.skipif(
    not os.environ.get("ARK_API_KEY"),
    reason="needs ARK_API_KEY for the live doubao responder",
)

# 16x16 PNG, base64 (content irrelevant — we only assert the call succeeds)
_TINY_PNG = (
    "iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAH0lEQVR42mNk+M9Qz0Bk"
    "YBxVSF+FjAxEglGFo0oBADjKA/3pPj7xAAAAAElFTkSuQmCC"
)


def test_to_openai_messages_converts_image_blocks():
    # Anthropic multimodal block → OpenAI image_url with a data: URI.
    anthropic = [
        {"role": "user", "content": "上一句"},  # plain string, untouched
        {"role": "user", "content": [
            {"type": "image", "source": {
                "type": "base64", "media_type": "image/jpeg", "data": "AAA"}},
            {"type": "text", "text": "看这张"},
        ]},
    ]
    out = ConversationEngine._to_openai_messages(anthropic)
    assert out[0] == {"role": "user", "content": "上一句"}
    parts = out[1]["content"]
    assert parts[0]["type"] == "image_url"
    assert parts[0]["image_url"]["url"] == "data:image/jpeg;base64,AAA"
    assert parts[1] == {"type": "text", "text": "看这张"}


@_needs_ark
@pytest.mark.asyncio
async def test_doubao_responder_generates_reply(sample_persona, mock_llm, tmp_path):
    sample_persona.responder.provider = "doubao"
    sample_persona.responder.model = "ep-REDACTED"
    engine = ConversationEngine(
        persona=sample_persona, llm_provider=mock_llm,
        memory_manager=MemoryManager(data_dir=str(tmp_path / "memory")),
    )
    assert engine._responder_is_external() is True
    out = await engine.chat_full("今天好累，啥也不想干")
    assert out.speech.strip()
    assert out.speech.strip() != "This is a mock response."  # doubao, not main
    print("\n[doubao reply]", out.speech)


@_needs_ark
@pytest.mark.asyncio
async def test_doubao_responder_streams_chunks(sample_persona, mock_llm, tmp_path):
    # The events path must emit incremental `chunk` events (live streaming),
    # not just a single buffered `done`.
    sample_persona.responder.provider = "doubao"
    sample_persona.responder.model = "ep-REDACTED"
    engine = ConversationEngine(
        persona=sample_persona, llm_provider=mock_llm,
        memory_manager=MemoryManager(data_dir=str(tmp_path / "memory")),
    )
    chunks, done = [], None
    async for ev in engine.chat_stream_events(
        "随便说点啥逗我开心", channel="feishu", recipient_id="t1"):
        if ev.type == "chunk":
            chunks.append(ev.content)
        elif ev.type == "done":
            done = ev.content
    assert len(chunks) >= 2, f"expected streamed chunks, got {len(chunks)}"
    assert done and done.strip()
    assert "===META===" not in "".join(chunks)  # delimiter never streamed
    print(f"\n[streamed {len(chunks)} chunks] done={done!r}")


@_needs_ark
@pytest.mark.asyncio
async def test_doubao_responder_handles_image_turn(sample_persona, mock_llm, tmp_path):
    # Image turn must NOT split off to Claude — it rides doubao via conversion.
    sample_persona.responder.provider = "doubao"
    sample_persona.responder.model = "ep-REDACTED"
    engine = ConversationEngine(
        persona=sample_persona, llm_provider=mock_llm,
        memory_manager=MemoryManager(data_dir=str(tmp_path / "memory")),
    )
    out = await engine.chat_full(
        "看这个", images=[{"media_type": "image/png", "data": _TINY_PNG}])
    assert out.speech.strip()
    assert out.speech.strip() != "This is a mock response."
    print("\n[doubao image reply]", out.speech)
