"""What she scans for, and whose voice it comes back in, follow the persona.

The fetch prompt named its categories (天文 / 文学 / 上海本地) and its writer
("Aria 的语气……她是 28 岁的天文人 + 写作者，住上海") in literal text. Running as
a school idol, that produced a daily feed of telescope launches and model
safety cards, written in the previous character's register — 200 stored facts
of it, and 4.33M input tokens spent to get them.
"""

from datetime import date

import pytest

from lingxi.persona.models import Identity, PersonaConfig
from lingxi.world.fetcher import build_fetch_prompt, fetch_daily_briefing


def _persona(**kw):
    return PersonaConfig(
        name="唐可可", id="tangkeke",
        identity=Identity(full_name="唐可可", age=18, occupation="学园偶像"),
        **kw)


class TestPrompt:
    def test_her_own_interests_are_what_gets_searched(self):
        p = _persona(world_interests=["Love Live / 偶像圈", "上海", "音乐 / 现场"])
        prompt = build_fetch_prompt(p, date(2026, 9, 7))

        assert "Love Live / 偶像圈" in prompt
        assert "音乐 / 现场" in prompt

    def test_the_previous_characters_beat_is_gone(self):
        p = _persona(world_interests=["Love Live / 偶像圈"])
        prompt = build_fetch_prompt(p, date(2026, 9, 7))

        for leftover in ("天文", "文学 / 出版", "天文人", "写作者"):
            assert leftover not in prompt, f"{leftover!r} is the old persona's"

    def test_the_rewrite_is_asked_for_in_this_personas_voice(self):
        p = _persona(world_interests=["上海"])
        prompt = build_fetch_prompt(p, date(2026, 9, 7))

        assert "唐可可" in prompt

    def test_the_date_reaches_the_prompt(self):
        p = _persona(world_interests=["上海"])
        assert "2026-09-07" in build_fetch_prompt(p, date(2026, 9, 7))

    def test_a_persona_with_no_interests_has_nothing_to_scan_for(self):
        assert build_fetch_prompt(_persona(), date(2026, 9, 7)) is None


class TestFetchSkips:
    """No interests means no call — that is where the 4.33M tokens went."""

    @pytest.mark.asyncio
    async def test_a_persona_with_no_interests_never_calls_the_model(self):
        class _LLM:
            def __init__(self):
                self.calls = 0

            async def complete(self, **kw):
                self.calls += 1
                raise AssertionError("must not be called")

        llm = _LLM()
        briefing = await fetch_daily_briefing(llm, _persona(), date(2026, 9, 7))

        assert llm.calls == 0
        assert briefing.is_empty()


class TestCategories:
    """Categories come from the persona, so the whitelist cannot be fixed."""

    @pytest.mark.asyncio
    async def test_a_persona_specific_category_survives(self):
        payload = (
            '{"items": [{"headline": "Liella! 新单曲", '
            '"voice": "新单曲下个月出 我已经在等了", '
            '"category": "Love Live / 偶像圈", "source": "lovelive.jp"}]}'
        )

        class _LLM:
            async def complete(self, **kw):
                return type("R", (), {"content": payload, "raw_content_blocks": []})()

        p = _persona(world_interests=["Love Live / 偶像圈"])
        briefing = await fetch_daily_briefing(_LLM(), p, date(2026, 9, 7))

        assert len(briefing.items) == 1
        assert briefing.items[0].category == "Love Live / 偶像圈"
        assert briefing.items[0].voice == "新单曲下个月出 我已经在等了"
