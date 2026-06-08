"""Integration: when responder.provider=doubao, the chat reply is generated
by a single coherent pass on the live doubao model (not the main LLM), routed
through the real engine path. Skips when ARK_API_KEY is absent."""

import os

import pytest

from lingxi.conversation.engine import ConversationEngine
from lingxi.memory.manager import MemoryManager


pytestmark = pytest.mark.skipif(
    not os.environ.get("ARK_API_KEY"),
    reason="needs ARK_API_KEY for the live doubao responder",
)


@pytest.mark.asyncio
async def test_doubao_responder_generates_reply(sample_persona, mock_llm, tmp_path):
    # Route the voice to the real doubao endpoint; orchestrator + bio stay on
    # the mock (offline). The mock would return a fixed English string, so a
    # Chinese reply that differs proves doubao actually produced the words.
    sample_persona.responder.provider = "doubao"
    sample_persona.responder.model = "ep-REDACTED"

    engine = ConversationEngine(
        persona=sample_persona,
        llm_provider=mock_llm,
        memory_manager=MemoryManager(data_dir=str(tmp_path / "memory")),
    )

    assert engine._responder_is_external() is True

    out = await engine.chat_full("今天好累，啥也不想干")
    speech = out.speech if hasattr(out, "speech") else str(out)

    assert speech.strip()
    assert speech.strip() != "This is a mock response."  # not the main/mock LLM
    print("\n[doubao reply]", speech)
