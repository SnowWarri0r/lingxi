"""Unit tests for ClaudeProvider prefill and top_p support (body shape only)."""

from lingxi.providers.claude import ClaudeProvider


def test_body_contains_top_p():
    provider = ClaudeProvider(api_key="sk-ant-api-test", model="claude-sonnet-4-20250514")
    body = provider._build_body(
        messages=[{"role": "user", "content": "hi"}],
        system=None,
        max_tokens=100,
        temperature=0.9,
        top_p=0.7,
    )
    assert body["top_p"] == 0.7
    assert body["temperature"] == 0.9


def test_body_omits_top_p_when_none():
    """When top_p is None (default), body should NOT contain the key — preserves API hygiene."""
    provider = ClaudeProvider(api_key="sk-ant-api-test", model="claude-sonnet-4-20250514")
    body = provider._build_body(
        messages=[{"role": "user", "content": "hi"}],
        system=None,
        max_tokens=100,
        temperature=0.9,
    )
    assert "top_p" not in body


def test_prefill_appends_assistant_message():
    provider = ClaudeProvider(api_key="sk-ant-api-test", model="claude-sonnet-4-20250514")
    messages = [{"role": "user", "content": "hi"}]
    body = provider._build_body(
        messages=messages,
        system=None,
        max_tokens=100,
        temperature=0.9,
        top_p=1.0,
        prefill="嗯",
    )
    assert body["messages"][-1] == {"role": "assistant", "content": "嗯"}


def test_empty_prefill_does_not_append():
    provider = ClaudeProvider(api_key="sk-ant-api-test", model="claude-sonnet-4-20250514")
    messages = [{"role": "user", "content": "hi"}]
    body = provider._build_body(
        messages=messages,
        system=None,
        max_tokens=100,
        temperature=0.9,
        top_p=1.0,
        prefill="",
    )
    assert body["messages"][-1]["role"] == "user"
