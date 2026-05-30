"""Tests for world.models."""

from datetime import date

import pytest

from lingxi.world.models import DailyBriefing, NewsItem


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
