"""The buffer keeps what matters; the prompt has to show all of it.

The short-term buffer spends her unanswered openers before his words, but
the prompt rendered only the newest 24 of its 30 turns. On the replay of
09-01..09-23 that meant 7 of the 9 things he had said that month reached
her, and the other two were kept on disk and never shown.
"""

from datetime import datetime, timedelta

from lingxi.conversation.context import ContextAssembler, TokenBudget
from lingxi.memory.manager import MemoryContext
from lingxi.memory.short_term import ConversationTurn

NOW = datetime(2026, 9, 23, 20, 0)


def _buffer():
    """His line first, then 29 of hers across a week — nothing inside 12h but the tail."""
    turns = [ConversationTurn(role="user", content="周末在家拼乐高",
                              timestamp=NOW - timedelta(days=7))]
    for i in range(29):
        turns.append(ConversationTurn(role="assistant", content=f"opener {i}",
                                      timestamp=NOW - timedelta(days=7) + timedelta(hours=5 * (i + 1))))
    return turns


def _rendered(budget):
    msgs = ContextAssembler(budget=budget).assemble_messages(
        MemoryContext(short_term_turns=_buffer()), now=NOW)
    return [m["content"] for m in msgs]


def test_the_oldest_turn_the_buffer_kept_is_shown():
    assert "周末在家拼乐高" in _rendered(TokenBudget())


def test_nothing_the_buffer_kept_is_reported_as_omitted():
    assert not any("省略了" in c for c in _rendered(TokenBudget()))


def test_the_default_matches_the_default_buffer():
    from lingxi.memory.short_term import ShortTermMemory
    assert TokenBudget().recent_turns_min >= ShortTermMemory().max_turns
