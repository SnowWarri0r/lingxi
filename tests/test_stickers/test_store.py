import pytest
from pathlib import Path

from lingxi.stickers.store import StickerStore
from lingxi.stickers.models import Sticker


async def _store(tmp_path) -> StickerStore:
    s = StickerStore(Path(tmp_path) / "stickers.db")
    await s.init()
    return s


@pytest.mark.asyncio
async def test_add_then_search_fts(tmp_path):
    s = await _store(tmp_path)
    await s.add(Sticker(
        file_path="/img/a.png", content_hash="h1",
        caption="无语翻白眼", emotion="无语",
        tags=["翻白眼", "无语"], when_to_use="对方说了离谱的话"))
    hits = await s.search("翻白眼", k=5)
    assert len(hits) == 1
    assert hits[0].caption == "无语翻白眼"


@pytest.mark.asyncio
async def test_search_short_query_like_fallback(tmp_path):
    # 2-char CJK query is below FTS5 trigram's 3-char minimum → LIKE fallback
    s = await _store(tmp_path)
    await s.add(Sticker(
        file_path="/img/b.png", content_hash="h2",
        caption="笑哭", emotion="好笑", tags=["笑哭"], when_to_use="觉得好笑"))
    hits = await s.search("笑哭", k=5)
    assert any(h.caption == "笑哭" for h in hits)


@pytest.mark.asyncio
async def test_add_dedupes_on_content_hash(tmp_path):
    s = await _store(tmp_path)
    first = await s.add(Sticker(
        file_path="/img/c.png", content_hash="dup", caption="A"))
    second = await s.add(Sticker(
        file_path="/img/c2.png", content_hash="dup", caption="B"))
    assert first is True       # inserted
    assert second is False     # skipped (same hash)
    hits = await s.search("A", k=5)
    assert len(hits) == 1
    assert hits[0].caption == "A"


@pytest.mark.asyncio
async def test_get_by_id(tmp_path):
    s = await _store(tmp_path)
    st = Sticker(file_path="/img/d.png", content_hash="h4", caption="比心")
    await s.add(st)
    got = await s.get(st.id)
    assert got is not None and got.caption == "比心"
