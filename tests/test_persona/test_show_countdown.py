"""A show known only by its month gets no day count.

Configured as 「2026-11」, the show was counted to November 1st: on 10-24
she was told 「还有 8 天」 for a show three weeks off, and when she
remembered the real date the prompt talked her out of it.
"""

from datetime import date
from types import SimpleNamespace as N

from lingxi.persona.prompt_builder import _upcoming_shows_block, build_stage_block


def _show(d):
    return N(event="Fes", date=d, venue="", note="")


def test_a_month_only_show_says_the_month_not_a_day_count():
    block = _upcoming_shows_block([_show("2026-11")], date(2026, 10, 24))
    assert "下个月，具体哪天还没定" in block
    assert "还有 8 天" not in block


def test_a_show_with_a_date_is_counted_to_that_date():
    block = _upcoming_shows_block([_show("2026-11-14")], date(2026, 10, 24))
    assert "还有 21 天" in block


def test_this_month_is_still_ahead_in_the_stage_block():
    persona = N(upcoming_shows=[_show("2026-11")], lexicon=[])
    block = build_stage_block(persona, date(2026, 11, 3))
    assert "就在这个月，具体哪天还没定" in block
