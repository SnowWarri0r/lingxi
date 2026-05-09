"""Tests for world.models + world.store."""

from datetime import date

import pytest

from lingxi.world.models import DailyBriefing, NewsItem
from lingxi.world.store import WorldStore


@pytest.fixture
def store(tmp_path):
    return WorldStore(tmp_path)


class TestModels:
    def test_empty_briefing_is_empty(self):
        b = DailyBriefing(date=date(2026, 5, 9))
        assert b.is_empty()

    def test_briefing_with_items_not_empty(self):
        b = DailyBriefing(
            date=date(2026, 5, 9),
            items=[NewsItem(headline="x", aria_voice="今早扫到 x", category="天文")],
        )
        assert not b.is_empty()

    def test_invalid_category_rejected(self):
        with pytest.raises(Exception):
            NewsItem(headline="x", aria_voice="y", category="不存在的类")


class TestStore:
    @pytest.mark.asyncio
    async def test_load_missing_returns_none(self, store):
        b = await store.load_for(date(2026, 5, 9))
        assert b is None

    @pytest.mark.asyncio
    async def test_save_and_load_roundtrip(self, store):
        b = DailyBriefing(
            date=date(2026, 5, 9),
            items=[NewsItem(
                headline="火星样本",
                aria_voice="今早扫到 NASA 说火星样本里有微生物迹象",
                category="天文",
            )],
        )
        await store.save(b)
        loaded = await store.load_for(date(2026, 5, 9))
        assert loaded is not None
        assert loaded.items[0].headline == "火星样本"
        assert loaded.items[0].category == "天文"

    @pytest.mark.asyncio
    async def test_has_briefing_for(self, store):
        d = date(2026, 5, 9)
        assert not await store.has_briefing_for(d)
        await store.save(DailyBriefing(date=d))
        assert await store.has_briefing_for(d)

    @pytest.mark.asyncio
    async def test_corrupted_file_returns_none(self, tmp_path, store):
        path = tmp_path / "world" / "news" / "2026-05-09.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{ this isn't json", encoding="utf-8")
        loaded = await store.load_for(date(2026, 5, 9))
        assert loaded is None

    @pytest.mark.asyncio
    async def test_load_today_with_data(self, store):
        from datetime import date as _d
        today = _d.today()
        await store.save(DailyBriefing(
            date=today,
            items=[NewsItem(headline="x", aria_voice="y")],
        ))
        b = await store.load_today()
        assert b is not None
        assert b.date == today
