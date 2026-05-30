"""Test that daily_briefing content never renders into the focus reminder.

Phase 3 context refactor removed _build_world_section and daily_briefing
rendering from prompt_builder. World news flows through brain/renderer
【身边的事】 block (FactStore world.* facts). The daily_briefing parameter
was removed entirely from build_turn_focus_reminder in P7.
"""

from lingxi.persona.models import Identity, PersonaConfig
from lingxi.persona.prompt_builder import PromptBuilder


def _persona():
    return PersonaConfig(name="T", identity=Identity(full_name="T"))


def _reminder(**kwargs):
    return PromptBuilder(_persona()).build_turn_focus_reminder(**kwargs)


class TestEmptyBriefingNoRender:
    def test_world_section_absent(self):
        # With no dynamic state, reminder returns None.
        reminder = _reminder()
        if reminder is not None:
            assert "今早扫到的事" not in reminder

    def test_focus_reminder_has_no_world_block(self):
        # Even with mood/time, world section is not present.
        reminder = _reminder(current_mood="开心")
        if reminder is not None:
            assert "今早扫到的事" not in reminder

    def test_world_content_never_surfaces(self):
        # Verify the reminder path produces no world news text.
        reminder = _reminder(current_mood="平静")
        if reminder is not None:
            assert "上海本地" not in reminder
            assert "今早扫到" not in reminder
