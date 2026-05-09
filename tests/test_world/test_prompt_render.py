"""Test that DailyBriefing renders into the focus reminder.

Phase 2 context refactor — daily_briefing is dynamic per-day state, lives
in `<system-reminder>` (recency channel), not the static system prompt.
"""

from datetime import date

from lingxi.inner_life.models import InnerState
from lingxi.persona.models import Identity, PersonaConfig
from lingxi.persona.prompt_builder import PromptBuilder
from lingxi.world.models import DailyBriefing, NewsItem


def _persona():
    return PersonaConfig(name="T", identity=Identity(full_name="T"))


def _reminder(**kwargs):
    return PromptBuilder(_persona()).build_turn_focus_reminder(**kwargs)


class TestEmptyBriefingNoRender:
    def test_none_briefing_no_section(self):
        reminder = _reminder(inner_state=InnerState(), daily_briefing=None)
        if reminder is not None:
            assert "今早扫到的事" not in reminder

    def test_empty_items_no_section(self):
        reminder = _reminder(
            inner_state=InnerState(),
            daily_briefing=DailyBriefing(date=date(2026, 5, 9), items=[]),
        )
        if reminder is not None:
            assert "今早扫到的事" not in reminder


class TestPopulatedRendering:
    def test_items_render_in_inner_state(self):
        b = DailyBriefing(
            date=date(2026, 5, 9),
            items=[
                NewsItem(
                    headline="JWST 火星观测",
                    aria_voice="今早扫到 JWST 拍到火星新数据",
                    category="天文",
                ),
                NewsItem(
                    headline="上海暴雨",
                    aria_voice="今天上海要下大雨",
                    category="上海本地",
                ),
            ],
        )
        reminder = _reminder(inner_state=InnerState(), daily_briefing=b)
        assert reminder is not None
        assert "今早扫到的事" in reminder
        assert "JWST" in reminder
        assert "今天上海要下大雨" in reminder
        assert "[天文]" in reminder
        assert "[上海本地]" in reminder

    def test_其他_category_no_tag(self):
        b = DailyBriefing(
            date=date(2026, 5, 9),
            items=[
                NewsItem(headline="x", aria_voice="一条普通的事", category="其他"),
            ],
        )
        reminder = _reminder(inner_state=InnerState(), daily_briefing=b)
        assert reminder is not None
        assert "[其他]" not in reminder
        assert "一条普通的事" in reminder

    def test_caps_at_5_items(self):
        b = DailyBriefing(
            date=date(2026, 5, 9),
            items=[
                NewsItem(headline=f"h{i}", aria_voice=f"voice-{i}", category="其他")
                for i in range(10)
            ],
        )
        reminder = _reminder(inner_state=InnerState(), daily_briefing=b)
        assert reminder is not None
        assert "voice-0" in reminder
        assert "voice-4" in reminder
        assert "voice-5" not in reminder


class TestStandaloneRender:
    def test_world_section_when_no_inner_state(self):
        # Even without inner_state, briefing should still surface
        b = DailyBriefing(
            date=date(2026, 5, 9),
            items=[NewsItem(headline="x", aria_voice="今天上海要下雨", category="上海本地")],
        )
        reminder = _reminder(inner_state=None, daily_briefing=b)
        assert reminder is not None
        assert "今早扫到" in reminder
        assert "今天上海要下雨" in reminder
