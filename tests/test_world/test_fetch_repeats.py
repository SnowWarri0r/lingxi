"""She should not scan the same news twice.

Nothing told the fetch what it had already brought back, so it re-reported
the same stories on later days: 「8th的会场和日程出来了 有明→福冈→名古屋」 on
09-08 and 「8th的日程出来了 有明→福冈→名古屋」 on 09-10, both at importance 6
— and the block pushes the single highest-importance item, so that one line
was in her head on two of three days. 体育祭 and 新视野号 repeated the same way.

Same shape as the opener path's 「你最近发过的主动消息（这次换一件事说）」.
"""

from datetime import date

from lingxi.persona.models import Identity, PersonaConfig
from lingxi.world.fetcher import build_fetch_prompt


def _persona():
    return PersonaConfig(
        name="唐可可", id="tangkeke",
        identity=Identity(full_name="唐可可", age=21),
        world_interests=["Love Live / 学园偶像圈"])


ALREADY = ["8th的会场和日程出来了 有明→福冈→名古屋 明年春天又要跑了",
           "体育祭昨天刚结束 还没缓过来"]


def test_what_she_already_scanned_is_in_the_prompt():
    prompt = build_fetch_prompt(_persona(), date(2026, 9, 10), recent=ALREADY)

    assert "8th的会场和日程出来了" in prompt
    assert "体育祭昨天刚结束" in prompt


def test_the_prompt_asks_for_something_she_has_not_seen():
    prompt = build_fetch_prompt(_persona(), date(2026, 9, 10), recent=ALREADY)

    assert "已经" in prompt


def test_a_first_run_has_nothing_to_exclude():
    prompt = build_fetch_prompt(_persona(), date(2026, 9, 10), recent=[])

    assert prompt is not None
    assert "8th" not in prompt


def test_recent_is_optional():
    """Callers that never pass it keep working."""
    assert build_fetch_prompt(_persona(), date(2026, 9, 10)) is not None


def test_blank_entries_are_dropped():
    prompt = build_fetch_prompt(_persona(), date(2026, 9, 10),
                                recent=["  ", "", "真的一条"])

    assert "真的一条" in prompt


def test_the_list_is_bounded():
    """Three days of scanning is context, not an archive."""
    many = [f"第{i}条新闻" for i in range(60)]
    prompt = build_fetch_prompt(_persona(), date(2026, 9, 10), recent=many)

    assert "第0条新闻" in prompt
    assert "第59条新闻" not in prompt
