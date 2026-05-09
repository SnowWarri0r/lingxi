"""Test the fetcher's response-parsing path without making real API calls.

The fetcher is expected to gracefully degrade on bad responses — these
tests exercise that path by mocking the Anthropic client.
"""

import json
from datetime import date
from unittest.mock import AsyncMock, patch

import pytest

from lingxi.world.fetcher import (
    _extract_text_from_blocks,
    _strip_json_fences,
    fetch_daily_briefing,
)


class TestParsingHelpers:
    def test_strip_fences_plain(self):
        assert _strip_json_fences('{"a":1}') == '{"a":1}'

    def test_strip_fences_json_marker(self):
        assert _strip_json_fences('```json\n{"a":1}\n```') == '{"a":1}'

    def test_strip_fences_no_lang(self):
        assert _strip_json_fences('```\n{"a":1}\n```') == '{"a":1}'

    def test_extract_text_from_dict_blocks(self):
        blocks = [
            {"type": "tool_use", "id": "x"},
            {"type": "text", "text": "first"},
            {"type": "tool_use", "id": "y"},
            {"type": "text", "text": "second"},
        ]
        result = _extract_text_from_blocks(blocks)
        assert "first" in result
        assert "second" in result

    def test_extract_text_skips_non_text(self):
        blocks = [{"type": "tool_use", "id": "x"}]
        assert _extract_text_from_blocks(blocks) == ""


@pytest.mark.asyncio
async def test_fetcher_parses_well_formed_response():
    payload = json.dumps({
        "items": [
            {
                "headline": "JWST 看到火星新数据",
                "aria_voice": "今早扫到 JWST 火星新数据",
                "category": "天文",
                "source": "nasa.gov",
                "url": "https://nasa.gov/example",
            },
        ],
    })

    class FakeResponse:
        content = [{"type": "text", "text": payload}]

    fake_messages = AsyncMock()
    fake_messages.create = AsyncMock(return_value=FakeResponse())
    fake_client = type("X", (), {"messages": fake_messages})()

    with patch("anthropic.AsyncAnthropic", return_value=fake_client):
        b = await fetch_daily_briefing("test-key", date(2026, 5, 9))

    assert len(b.items) == 1
    assert b.items[0].category == "天文"
    assert "JWST" in b.items[0].headline


@pytest.mark.asyncio
async def test_fetcher_returns_empty_on_garbage_response():
    class FakeResponse:
        content = [{"type": "text", "text": "not json at all"}]

    fake_messages = AsyncMock()
    fake_messages.create = AsyncMock(return_value=FakeResponse())
    fake_client = type("X", (), {"messages": fake_messages})()

    with patch("anthropic.AsyncAnthropic", return_value=fake_client):
        b = await fetch_daily_briefing("test-key", date(2026, 5, 9))

    assert b.is_empty()


@pytest.mark.asyncio
async def test_fetcher_returns_empty_on_api_error():
    fake_messages = AsyncMock()
    fake_messages.create = AsyncMock(side_effect=RuntimeError("network down"))
    fake_client = type("X", (), {"messages": fake_messages})()

    with patch("anthropic.AsyncAnthropic", return_value=fake_client):
        b = await fetch_daily_briefing("test-key", date(2026, 5, 9))

    assert b.is_empty()


@pytest.mark.asyncio
async def test_fetcher_invalid_category_falls_to_其他():
    payload = json.dumps({
        "items": [{
            "headline": "x", "aria_voice": "y",
            "category": "garbage_category",
        }],
    })

    class FakeResponse:
        content = [{"type": "text", "text": payload}]

    fake_messages = AsyncMock()
    fake_messages.create = AsyncMock(return_value=FakeResponse())
    fake_client = type("X", (), {"messages": fake_messages})()

    with patch("anthropic.AsyncAnthropic", return_value=fake_client):
        b = await fetch_daily_briefing("test-key", date(2026, 5, 9))

    assert b.items[0].category == "其他"


@pytest.mark.asyncio
async def test_fetcher_skips_items_missing_required_fields():
    payload = json.dumps({
        "items": [
            {"headline": "ok", "aria_voice": "yes"},
            {"headline": "", "aria_voice": "no headline"},        # skip
            {"headline": "no voice", "aria_voice": ""},           # skip
            "not a dict",                                          # skip
        ],
    })

    class FakeResponse:
        content = [{"type": "text", "text": payload}]

    fake_messages = AsyncMock()
    fake_messages.create = AsyncMock(return_value=FakeResponse())
    fake_client = type("X", (), {"messages": fake_messages})()

    with patch("anthropic.AsyncAnthropic", return_value=fake_client):
        b = await fetch_daily_briefing("test-key", date(2026, 5, 9))

    assert len(b.items) == 1
    assert b.items[0].headline == "ok"
