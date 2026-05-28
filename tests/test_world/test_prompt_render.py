"""Test that DailyBriefing no longer renders into the focus reminder.

Phase 3 context refactor — _build_world_section and the daily_briefing
rendering inside _build_inner_state_section were removed from prompt_builder.
World news now flows through brain/renderer 【身边的事】 block (FactStore world.*
facts) on the _prepare_turn_v2 path. The daily_briefing param is silently ignored.
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

    def test_populated_briefing_also_not_rendered(self):
        # World section removed — even populated DailyBriefing is silently ignored.
        # Rendering is now brain/renderer's job via 【身边的事】.
        b = DailyBriefing(
            date=date(2026, 5, 9),
            items=[NewsItem(headline="x", aria_voice="今天上海要下雨", category="上海本地")],
        )
        reminder = _reminder(inner_state=InnerState(), daily_briefing=b)
        if reminder is not None:
            assert "今早扫到的事" not in reminder
