"""Chat history carries date dividers.

Regression: turns were rendered as bare {role, content} with the timestamp
dropped, so the whole buffer read as one continuous recent stretch. A remark
made on the 7th was followed up as '昨天说的' on the 11th — four days and three
follow-ups later, each follow-up re-reading the previous one as fresh.
"""
from datetime import datetime, timedelta

from lingxi.conversation.context import ContextAssembler, _day_label
from lingxi.memory.manager import MemoryContext
from lingxi.memory.short_term import ConversationTurn


# _day_label takes `now` explicitly, so it can be pinned. assemble_messages
# reads the real clock, so turns fed to it are built relative to real now —
# a frozen base there silently rots the day-gap assertions overnight.
PINNED = datetime(2026, 8, 11, 10, 0)
NOW = datetime.now()


def test_day_label_marks_today_yesterday_and_the_gap():
    assert "今天" in _day_label(PINNED.date(), PINNED)
    assert "昨天" in _day_label((PINNED - timedelta(days=1)).date(), PINNED)
    older = _day_label((PINNED - timedelta(days=4)).date(), PINNED)
    assert "4天前" in older          # the gap is spelled out, not inferred
    assert "8月7日" in older and "周五" in older


def _turn(role, content, when):
    return ConversationTurn(role=role, content=content, timestamp=when)


def test_divider_inserted_once_per_day_in_order():
    turns = [
        _turn("user", "腰已经酸了", NOW - timedelta(days=4)),
        _turn("assistant", "起来动动嘛", NOW - timedelta(days=4, minutes=-5)),
        _turn("assistant", "腰好点没", NOW - timedelta(days=1)),
        _turn("user", "好多了", NOW),
    ]
    msgs = ContextAssembler().assemble_messages(MemoryContext(short_term_turns=turns))
    dividers = [m["content"] for m in msgs if m["content"].startswith("[——")]
    assert len(dividers) == 3                       # three distinct days
    assert "4天前" in dividers[0]
    assert "昨天" in dividers[1]
    assert "今天" in dividers[2]
    # content itself is untouched and stays in order
    body = [m["content"] for m in msgs if not m["content"].startswith("[——")]
    assert body == ["腰已经酸了", "起来动动嘛", "腰好点没", "好多了"]


def test_same_day_turns_share_one_divider():
    turns = [
        _turn("user", "早", NOW - timedelta(hours=3)),
        _turn("assistant", "早呀", NOW - timedelta(hours=2)),
        _turn("user", "在忙吗", NOW),
    ]
    msgs = ContextAssembler().assemble_messages(MemoryContext(short_term_turns=turns))
    assert sum(1 for m in msgs if m["content"].startswith("[——")) == 1
