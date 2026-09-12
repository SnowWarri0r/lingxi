"""Tests for world.models."""

from datetime import date


from lingxi.world.models import DailyBriefing, NewsItem


class TestModels:
    def test_empty_briefing_is_empty(self):
        b = DailyBriefing(date=date(2026, 5, 9))
        assert b.is_empty()

    def test_briefing_with_items_not_empty(self):
        b = DailyBriefing(
            date=date(2026, 5, 9),
            items=[NewsItem(headline="x", voice="今早扫到 x", category="天文")],
        )
        assert not b.is_empty()

    def test_any_category_is_accepted(self):
        """Categories are the persona's own world_interests — no fixed set.

        The old Literal listed 天文 / 文学 / 上海本地, which is one character's
        reading list; a school idol's categories were all coerced to 其他.
        """
        item = NewsItem(headline="x", voice="y", category="Love Live / 偶像圈")
        assert item.category == "Love Live / 偶像圈"
