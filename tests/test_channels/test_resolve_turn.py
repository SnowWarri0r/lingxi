"""Regression: _resolve_turn must enforce recipient match + reject glob meta-chars."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from lingxi.channels.feishu import FeishuBot
from lingxi.fewshot.models import AnnotationTurn


def _make_bot_with_turns(turns: list[AnnotationTurn], turns_dir):
    """Build a minimally-configured FeishuBot proxy that satisfies _resolve_turn deps."""
    bot = FeishuBot.__new__(FeishuBot)
    engine = MagicMock()
    store = MagicMock()
    store.turns_dir = str(turns_dir)
    by_id = {t.turn_id: t for t in turns}

    async def _get(turn_id):
        return by_id.get(turn_id)

    store.get_turn = AsyncMock(side_effect=_get)
    engine.annotation_store = store
    bot.engine = engine
    return bot


@pytest.fixture
def turns_dir(tmp_path):
    d = tmp_path / "turns"
    d.mkdir()
    return d


def _seed_turn_files(turns_dir, turns):
    for t in turns:
        (turns_dir / f"{t.turn_id}.json").write_text("{}", encoding="utf-8")


@pytest.mark.asyncio
async def test_resolve_rejects_glob_metachars(turns_dir):
    turns = [
        AnnotationTurn(
            turn_id="abc12345-0000-0000-0000-000000000001",
            recipient_key="feishu:chat_A",
            user_message="hi",
            inner_thought="t",
            speech="s",
        ),
    ]
    _seed_turn_files(turns_dir, turns)
    bot = _make_bot_with_turns(turns, turns_dir)

    # Glob meta-chars must be rejected outright (not used as glob pattern)
    assert await bot._resolve_turn("abc?2345") is None
    assert await bot._resolve_turn("abc[1-5]") is None
    assert await bot._resolve_turn("*") is None
    assert await bot._resolve_turn("../etc") is None


@pytest.mark.asyncio
async def test_resolve_accepts_valid_uuid_prefix(turns_dir):
    turns = [
        AnnotationTurn(
            turn_id="abc12345-0000-0000-0000-000000000001",
            recipient_key="feishu:chat_A",
            user_message="hi",
            inner_thought="t",
            speech="s",
            created_at=datetime.now(),
        ),
    ]
    _seed_turn_files(turns_dir, turns)
    bot = _make_bot_with_turns(turns, turns_dir)

    found = await bot._resolve_turn("abc12345")
    assert found is not None
    assert found.turn_id == turns[0].turn_id


@pytest.mark.asyncio
async def test_resolve_rejects_wrong_recipient(turns_dir):
    """Critical: A user can't access B's turn even if they guess the prefix."""
    turns = [
        AnnotationTurn(
            turn_id="bbb22222-0000-0000-0000-000000000002",
            recipient_key="feishu:chat_B",
            user_message="hi",
            inner_thought="user B's thought",
            speech="s",
        ),
    ]
    _seed_turn_files(turns_dir, turns)
    bot = _make_bot_with_turns(turns, turns_dir)

    # User in chat_A tries to fetch B's turn
    found = await bot._resolve_turn(
        "bbb22222", expected_recipient="feishu:chat_A"
    )
    assert found is None, "B's turn must not be visible to A"

    # B itself can fetch
    found = await bot._resolve_turn(
        "bbb22222", expected_recipient="feishu:chat_B"
    )
    assert found is not None
