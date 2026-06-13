"""Sticker sending on the no-tools (doubao) path: the persona puts a sticker
intent in the ===META=== block; the engine searches the store and stages the
top match for the turn-end emit."""

import pytest

from lingxi.conversation.engine import ConversationEngine
from lingxi.conversation.output_schema import parse_turn_output
from lingxi.memory.manager import MemoryManager


def test_parse_extracts_sticker_field():
    raw = '哇 小鱼干\n===META===\n{"mood":"开心", "sticker":"开心打滚"}'
    out = parse_turn_output(raw)
    assert out.speech.strip() == "哇 小鱼干"
    assert out.sticker == "开心打滚"


def test_parse_sticker_defaults_empty():
    out = parse_turn_output('在呀\n===META===\n{"mood":"平静"}')
    assert out.sticker == ""


class _FakeStickerStore:
    def __init__(self, hits):
        self._hits = hits

    async def search(self, query, k=5):
        return self._hits


class _Hit:
    def __init__(self, path):
        self.file_path = path


def _engine(sample_persona, mock_llm, tmp_path, store):
    return ConversationEngine(
        persona=sample_persona, llm_provider=mock_llm,
        memory_manager=MemoryManager(data_dir=str(tmp_path / "m")),
        sticker_store=store,
    )


@pytest.mark.asyncio
async def test_resolve_sticker_stages_top_hit(sample_persona, mock_llm, tmp_path):
    store = _FakeStickerStore([_Hit("/x/happy.gif"), _Hit("/x/other.gif")])
    eng = _engine(sample_persona, mock_llm, tmp_path, store)
    await eng._resolve_sticker("开心打滚", "feishu:t1")
    assert eng._pending_stickers["feishu:t1"] == "/x/happy.gif"  # top hit, not random


@pytest.mark.asyncio
async def test_resolve_sticker_noop_on_no_match(sample_persona, mock_llm, tmp_path):
    eng = _engine(sample_persona, mock_llm, tmp_path, _FakeStickerStore([]))
    await eng._resolve_sticker("无语", "feishu:t1")
    assert not eng._pending_stickers.get("feishu:t1")


@pytest.mark.asyncio
async def test_resolve_sticker_noop_without_store(sample_persona, mock_llm, tmp_path):
    eng = _engine(sample_persona, mock_llm, tmp_path, None)
    await eng._resolve_sticker("开心", "feishu:t1")  # must not raise
    assert not eng._pending_stickers.get("feishu:t1")
