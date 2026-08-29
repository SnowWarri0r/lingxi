"""What happens to the buffer when generation comes back empty.

The fallback line is shown to the user so the card never hangs, but it is a
technical notice, not something she said. Writing it into the dialogue history
meant she read her own stock apology back as context on the next turn — and it
buried the user's message, which nobody had actually answered yet.
"""

import pytest

from lingxi.conversation.engine import EMPTY_REPLY_FALLBACK, ConversationEngine
from lingxi.memory.manager import MemoryManager


class _DeadResponder:
    """Stands in for the live 400/402 failures (no image support, no balance)."""

    async def complete_stream(self, **kwargs):
        raise RuntimeError("Error code: 400 - This model does not support image")
        yield  # pragma: no cover — makes this an async generator

    async def complete(self, **kwargs):
        raise RuntimeError("Error code: 400 - This model does not support image")


def _engine(persona, llm, tmp_path):
    persona.responder.provider = "deepseek"
    eng = ConversationEngine(
        persona=persona, llm_provider=llm,
        memory_manager=MemoryManager(data_dir=str(tmp_path / "memory")),
    )
    eng._responder_llm = _DeadResponder()
    return eng


async def _run(eng, text="在吗"):
    return [e async for e in eng.chat_stream_events(
        text, channel="feishu", recipient_id="oc_test")]


@pytest.mark.asyncio
async def test_the_user_still_gets_a_reply(sample_persona, mock_llm, tmp_path):
    events = await _run(_engine(sample_persona, mock_llm, tmp_path))
    done = [e for e in events if e.type == "done"]
    assert done and done[0].content == EMPTY_REPLY_FALLBACK


@pytest.mark.asyncio
async def test_the_fallback_is_not_written_to_history(
        sample_persona, mock_llm, tmp_path):
    eng = _engine(sample_persona, mock_llm, tmp_path)
    await _run(eng)

    history = eng.memory.short_term.get_history()
    assert not any(t.role == "assistant" for t in history)


@pytest.mark.asyncio
async def test_the_users_message_stays_in_history_unanswered(
        sample_persona, mock_llm, tmp_path):
    """She gets a second chance at it next turn instead of losing it."""
    eng = _engine(sample_persona, mock_llm, tmp_path)
    await _run(eng, "你看这个")

    history = eng.memory.short_term.get_history()
    assert history[-1].role == "user"
    assert "你看这个" in history[-1].content


@pytest.mark.asyncio
async def test_a_real_reply_is_still_remembered(sample_persona, mock_llm, tmp_path):
    class _LiveResponder:
        async def complete_stream(self, **kwargs):
            class _C:
                content = "在的在的"
            yield _C()

    eng = _engine(sample_persona, mock_llm, tmp_path)
    eng._responder_llm = _LiveResponder()
    await _run(eng)

    history = eng.memory.short_term.get_history()
    assert history[-1].role == "assistant"
    assert "在的在的" in history[-1].content
