"""Test the fetcher's response-parsing path without making real API calls.

The fetcher is expected to gracefully degrade on bad responses — these
tests exercise that path by mocking the Anthropic client.
"""

import json
from datetime import date
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from lingxi.persona.models import Identity, PersonaConfig
from lingxi.world.fetcher import (
    _extract_text_from_blocks,
    _strip_json_fences,
    fetch_daily_briefing,
)


def _persona():
    """Any persona with interests — these tests are about parsing, not topics."""
    return PersonaConfig(
        name="唐可可", id="tangkeke",
        identity=Identity(full_name="唐可可", age=18),
        world_interests=["Love Live / 偶像圈"])


def _fake_llm(text: str = "", *, error: Exception | None = None):
    """A stand-in LLM provider whose .complete() returns the given text (as
    result.content) or raises. The fetcher now routes through the shared
    provider, so we mock that instead of the raw Anthropic SDK."""
    complete = AsyncMock(
        side_effect=error if error is not None else None,
        return_value=SimpleNamespace(content=text, raw_content_blocks=[]),
    )
    return SimpleNamespace(complete=complete)


class TestParsingHelpers:
    def test_strip_fences_plain(self):
        assert _strip_json_fences('{"a":1}') == '{"a":1}'

    def test_strip_fences_json_marker(self):
        assert _strip_json_fences('```json\n{"a":1}\n```') == '{"a":1}'

    def test_strip_fences_no_lang(self):
        assert _strip_json_fences('```\n{"a":1}\n```') == '{"a":1}'

    def test_strip_fences_prose_preamble(self):
        # web_search replies preface the JSON with explanation text.
        raw = 'Based on my searches, here it is:\n\n```json\n{"a":1}\n```'
        assert _strip_json_fences(raw) == '{"a":1}'

    def test_strip_fences_bare_object_with_preamble(self):
        # No fence, just prose then a JSON object.
        assert _strip_json_fences('Here you go: {"a":1} done') == '{"a":1}'

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

    b = await fetch_daily_briefing(_fake_llm(payload), _persona(), date(2026, 5, 9))

    assert len(b.items) == 1
    assert b.items[0].category == "天文"
    assert "JWST" in b.items[0].headline


@pytest.mark.asyncio
async def test_fetcher_returns_empty_on_garbage_response():
    b = await fetch_daily_briefing(_fake_llm("not json at all"), _persona(), date(2026, 5, 9))

    assert b.is_empty()


@pytest.mark.asyncio
async def test_fetcher_returns_empty_on_api_error():
    llm = _fake_llm(error=RuntimeError("network down"))
    b = await fetch_daily_briefing(llm, _persona(), date(2026, 5, 9))

    assert b.is_empty()


@pytest.mark.asyncio
async def test_fetcher_keeps_the_personas_own_category():
    """No whitelist: the categories asked for are the persona's interests."""
    payload = json.dumps({
        "items": [{
            "headline": "x", "voice": "y",
            "category": "Love Live / 偶像圈",
        }],
    })

    b = await fetch_daily_briefing(_fake_llm(payload), _persona(), date(2026, 5, 9))

    assert b.items[0].category == "Love Live / 偶像圈"


@pytest.mark.asyncio
async def test_fetcher_bounds_a_runaway_category():
    payload = json.dumps({
        "items": [{"headline": "x", "voice": "y", "category": "категория" * 40}],
    })

    b = await fetch_daily_briefing(_fake_llm(payload), _persona(), date(2026, 5, 9))

    assert len(b.items[0].category) <= 40


@pytest.mark.asyncio
async def test_fetcher_still_reads_the_old_key():
    """A cached prompt or older reply may still say aria_voice."""
    payload = json.dumps({"items": [{"headline": "x", "aria_voice": "旧键"}]})

    b = await fetch_daily_briefing(_fake_llm(payload), _persona(), date(2026, 5, 9))

    assert b.items[0].voice == "旧键"


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

    b = await fetch_daily_briefing(_fake_llm(payload), _persona(), date(2026, 5, 9))

    assert len(b.items) == 1
    assert b.items[0].headline == "ok"
