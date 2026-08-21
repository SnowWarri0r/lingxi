"""The responder's calls must reach the debug log.

It was the one provider that never logged, and the gap only surfaced weeks
later: a failure measured live could not be reproduced afterwards, and with
no record of what had been sent or returned there was nothing to diff.
"""

import json

import pytest

from lingxi.providers.openai_provider import OpenAIProvider


class _Msg:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Msg(content)
        self.finish_reason = "stop"


class _Usage:
    prompt_tokens = 100
    completion_tokens = 20
    prompt_cache_hit_tokens = 64


class _Resp:
    model = "deepseek-v4-flash"
    usage = _Usage()

    def __init__(self, content):
        self.choices = [_Choice(content)]


class _Delta:
    def __init__(self, content):
        self.content = content


class _StreamChoice:
    def __init__(self, content):
        self.delta = _Delta(content)


class _StreamChunk:
    usage = None

    def __init__(self, content):
        self.choices = [_StreamChoice(content)]


def _provider(monkeypatch, response):
    p = OpenAIProvider(api_key="k", model="deepseek-v4-flash")

    class _Completions:
        async def create(self, **kwargs):
            if kwargs.get("stream"):
                async def _gen():
                    for piece in response:
                        yield _StreamChunk(piece)
                return _gen()
            return _Resp(response)

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()

    monkeypatch.setattr(p, "_get_client", lambda: _Client())
    return p


def _records(tmp_path):
    files = list((tmp_path / "debug" / "llm_requests").glob("*.jsonl"))
    if not files:
        return []
    return [json.loads(line) for line in files[0].read_text().splitlines()]


@pytest.fixture
def logging_on(tmp_path, monkeypatch):
    monkeypatch.setenv("LINGXI_DEBUG_LLM", "1")
    monkeypatch.setenv("MEMORY_DATA_DIR", str(tmp_path / "memory"))
    return tmp_path


@pytest.mark.asyncio
async def test_complete_is_logged(logging_on, monkeypatch):
    p = _provider(monkeypatch, "诶嘿嘿")
    await p.complete(messages=[{"role": "user", "content": "在吗"}],
                     system="你是唐可可", _debug_purpose="chat_test")

    rec = _records(logging_on)
    assert len(rec) == 1
    assert rec[0]["response"] == "诶嘿嘿"
    assert rec[0]["system"] == "你是唐可可"
    assert rec[0]["purpose"] == "chat_test"
    assert rec[0]["model"] == "deepseek-v4-flash"


@pytest.mark.asyncio
async def test_stream_logs_the_assembled_reply(logging_on, monkeypatch):
    """A streamed turn is what the user actually reads — log the whole thing,
    not the fragments it arrived in."""
    p = _provider(monkeypatch, ["诶", "嘿", "嘿"])
    out = ""
    async for chunk in p.complete_stream(
        messages=[{"role": "user", "content": "在吗"}], system="你是唐可可"):
        out += chunk.content

    rec = _records(logging_on)
    assert len(rec) == 1
    assert rec[0]["response"] == "诶嘿嘿" == out


@pytest.mark.asyncio
async def test_nothing_is_written_when_logging_is_off(tmp_path, monkeypatch):
    monkeypatch.delenv("LINGXI_DEBUG_LLM", raising=False)
    monkeypatch.setenv("MEMORY_DATA_DIR", str(tmp_path / "memory"))
    p = _provider(monkeypatch, "诶嘿嘿")
    await p.complete(messages=[{"role": "user", "content": "在吗"}])

    assert _records(tmp_path) == []


@pytest.mark.asyncio
async def test_a_logging_failure_does_not_break_the_turn(logging_on, monkeypatch):
    """Debug logging is never allowed to cost a reply."""
    import lingxi.debug.request_log as rl

    def _boom(**kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(rl, "log_request", _boom)
    p = _provider(monkeypatch, "诶嘿嘿")
    result = await p.complete(messages=[{"role": "user", "content": "在吗"}])

    assert result.content == "诶嘿嘿"
