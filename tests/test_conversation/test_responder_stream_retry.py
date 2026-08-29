"""The streamed reply gets one more attempt — but only when that can help.

A dropped connection used to cost the whole turn: the user got the empty-reply
fallback for something that would have worked a second later. Retrying every
failure is the opposite bug, and retrying after text is already on the card
would repeat it.
"""

import pytest

from lingxi.conversation.engine import EMPTY_REPLY_FALLBACK, ConversationEngine
from lingxi.memory.manager import MemoryManager


class _Chunk:
    def __init__(self, content):
        self.content = content


class _Responder:
    """Fails the first attempt with `exc`, then streams `text`."""

    def __init__(self, exc, text="在的在的", fail_after=""):
        self._exc = exc
        self._text = text
        self._fail_after = fail_after
        self.attempts = 0

    async def complete_stream(self, **kwargs):
        self.attempts += 1
        if self.attempts == 1:
            if self._fail_after:
                yield _Chunk(self._fail_after)
            raise self._exc
        yield _Chunk(self._text)


def _status(code):
    exc = RuntimeError(f"Error code: {code}")
    exc.status_code = code
    return exc


def _engine(persona, llm, tmp_path, responder):
    persona.responder.provider = "deepseek"
    eng = ConversationEngine(
        persona=persona, llm_provider=llm,
        memory_manager=MemoryManager(data_dir=str(tmp_path / "memory")),
    )
    eng._responder_llm = responder
    return eng


async def _speech(eng):
    events = [e async for e in eng.chat_stream_events(
        "在吗", channel="feishu", recipient_id="oc_test")]
    return next(e.content for e in events if e.type == "done")


@pytest.mark.asyncio
async def test_a_transient_failure_is_retried_and_the_turn_survives(
        sample_persona, mock_llm, tmp_path):
    r = _Responder(_status(503))
    eng = _engine(sample_persona, mock_llm, tmp_path, r)

    assert "在的在的" in await _speech(eng)
    assert r.attempts == 2


@pytest.mark.asyncio
async def test_a_dropped_connection_is_retried(sample_persona, mock_llm, tmp_path):
    import openai

    r = _Responder(openai.APIConnectionError(request=None))
    eng = _engine(sample_persona, mock_llm, tmp_path, r)

    assert "在的在的" in await _speech(eng)
    assert r.attempts == 2


@pytest.mark.asyncio
async def test_a_rejected_request_is_not_sent_twice(
        sample_persona, mock_llm, tmp_path):
    """400 image / 402 balance — the second call fails identically and bills."""
    r = _Responder(_status(400))
    eng = _engine(sample_persona, mock_llm, tmp_path, r)

    assert await _speech(eng) == EMPTY_REPLY_FALLBACK
    assert r.attempts == 1


@pytest.mark.asyncio
async def test_an_empty_account_is_not_charged_twice(
        sample_persona, mock_llm, tmp_path):
    r = _Responder(_status(402))
    eng = _engine(sample_persona, mock_llm, tmp_path, r)

    assert await _speech(eng) == EMPTY_REPLY_FALLBACK
    assert r.attempts == 1


@pytest.mark.asyncio
async def test_text_already_on_screen_is_never_restarted(
        sample_persona, mock_llm, tmp_path):
    """A retry would restart from scratch, repeating what the card already shows."""
    r = _Responder(_status(503), fail_after="诶我在的呀，刚才在弹钢琴")
    eng = _engine(sample_persona, mock_llm, tmp_path, r)

    speech = await _speech(eng)
    assert r.attempts == 1
    assert speech.count("诶我在的呀") == 1, "the partial reply is kept, not doubled"


@pytest.mark.asyncio
async def test_a_healthy_call_costs_one_attempt(sample_persona, mock_llm, tmp_path):
    class _Fine:
        attempts = 0

        async def complete_stream(self, **kwargs):
            _Fine.attempts += 1
            yield _Chunk("在的在的")

    eng = _engine(sample_persona, mock_llm, tmp_path, _Fine())
    assert "在的在的" in await _speech(eng)
    assert _Fine.attempts == 1
