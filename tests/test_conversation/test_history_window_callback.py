"""A callback to something said days ago has to find it still there.

This conversation runs in bursts a few days apart, not daily. On the live
buffer the assembler held 27 turns and put 6 into the prompt — the other 21
dropped by the session window, not by the token budget, which was barely
touched. So 「写好了！比我想的顺利」, referring to a letter discussed two days
and eight turns earlier, arrived with 「[省略了 6 轮较早的对话]」 where the
letter should have been. She answered it warmly and said nothing.

Scored sampling could not separate that from the healthy case — she echoes
the user's own words either way — so the guard is here, where it is exact.
"""

from datetime import datetime, timedelta

from lingxi.conversation.context import ContextAssembler, TokenBudget, estimate_tokens
from lingxi.memory.manager import MemoryContext
from lingxi.memory.short_term import ConversationTurn


NOW = datetime(2026, 9, 2, 21, 30)


def _turns(*specs) -> list[ConversationTurn]:
    return [
        ConversationTurn(role=role, content=content,
                         timestamp=NOW - timedelta(minutes=mins))
        for role, content, mins in specs
    ]


# The letter, two days back; then a full evening of unrelated chat on top.
BURST = _turns(
    ("user", "我想给她写封信 手写的那种", 2900),
    ("assistant", "手写的好呀 比打印出来的有分量", 2898),
    ("user", "但我不知道写什么 怕写得太肉麻", 2890),
    ("assistant", "肉麻怕啥 写你最想让她知道的那件事", 2888),
    *[(r, f"今天的第 {i} 句闲聊", m)
      for i, (r, m) in enumerate(
          [("user", 300), ("assistant", 298), ("user", 250), ("assistant", 248),
           ("user", 180), ("assistant", 178), ("user", 60), ("assistant", 58)])],
)


def _rendered(turns, budget=None) -> str:
    messages = ContextAssembler(budget=budget).assemble_messages(
        MemoryContext(short_term_turns=turns), now=NOW)
    return "\n".join(m["content"] for m in messages)


def test_the_referent_survives_a_burst_of_newer_chat():
    assert "怕写得太肉麻" in _rendered(BURST)


def test_the_old_setting_would_have_dropped_it():
    """Pins the regression, and what the size of the window is actually for."""
    six = TokenBudget(recent_turns_min=6)

    assert "怕写得太肉麻" not in _rendered(BURST, six)
    assert "省略了" in _rendered(BURST, six)


def test_nothing_is_omitted_when_everything_fits():
    assert "省略了" not in _rendered(BURST)


def test_the_newest_turns_are_always_present():
    rendered = _rendered(BURST)
    assert "今天的第 7 句闲聊" in rendered


def test_a_long_history_still_stops_at_the_budget():
    """The guarantee is not a licence to send everything ever said."""
    many = _turns(*[("user", f"第 {i} 句" * 40, 5000 - i) for i in range(200)])

    rendered = _rendered(many)

    assert estimate_tokens(rendered) < TokenBudget().history_budget * 2
    assert "省略了" in rendered, "a 200-turn history must still be trimmed"


def test_date_dividers_still_mark_the_gap():
    """Two days of silence must not read as one continuous conversation."""
    rendered = _rendered(BURST)
    assert "——]" in rendered
