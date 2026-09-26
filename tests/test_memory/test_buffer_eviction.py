"""Her own unanswered messages were deciding what she forgot about him.

The short-term buffer holds 30 turns and evicted by age alone. Proactive
openers arrive about three a day whether he answers or not, so on 2026-09-23
his buffer covered six days and held 24 of her turns to 6 of his; everything
he had said before 09-17 was gone. Nothing else held it — episode summaries
were retired, thread_summary is in memory only, and no fact about him had
been written since 08-28. Replaying that month under the old rule reproduced
the live buffer exactly, and projected forward it forgot a thing she was
still asking him about after six days of silence.

A run of messages she sent with no reply between them now gives up its older
members first, keeping the last two — the one he answers, and one before it.
"""

from datetime import datetime, timedelta

import pytest

from lingxi.memory.short_term import ConversationTurn, ShortTermMemory, trim_to_cap

T0 = datetime(2026, 9, 1, 8, 0)


def _t(role, text, k):
    return ConversationTurn(role=role, content=text, timestamp=T0 + timedelta(hours=k))


def _his(turns):
    return [t.content for t in turns if t.role == "user"]


class TestTheRule:
    def test_below_the_cap_nothing_is_touched(self):
        turns = [_t("assistant", f"o{i}", i) for i in range(10)]
        assert trim_to_cap(turns, 30) == turns

    def test_a_run_of_her_openers_goes_before_his_words(self):
        turns = [_t("user", "周末在家拼乐高", 0), _t("assistant", "reply", 1)]
        turns += [_t("assistant", f"opener {i}", 2 + i) for i in range(5)]

        kept = trim_to_cap(turns, 4)

        assert "周末在家拼乐高" in _his(kept)

    def test_the_last_two_of_a_run_survive(self):
        """The one he eventually answers must still be there."""
        turns = [_t("user", "hi", 0)] + [_t("assistant", f"opener {i}", 1 + i) for i in range(6)]

        kept = [t.content for t in trim_to_cap(turns, 3)]

        assert kept == ["hi", "opener 4", "opener 5"]

    def test_an_ordinary_exchange_still_evicts_oldest_first(self):
        """No runs to spend: the old behaviour, exactly."""
        turns = []
        for i in range(20):
            turns += [_t("user", f"u{i}", 2 * i), _t("assistant", f"a{i}", 2 * i + 1)]

        assert trim_to_cap(turns, 30) == turns[-30:]

    def test_order_is_preserved(self):
        turns = [_t("user", "u", 0)] + [_t("assistant", f"o{i}", 1 + i) for i in range(8)] \
            + [_t("user", "v", 20)]

        kept = trim_to_cap(turns, 5)

        assert [t.timestamp for t in kept] == sorted(t.timestamp for t in kept)
        assert len(kept) == 5


class TestTheMonthThatHappened:
    """Shape of 09-01..09-23: nine of his turns, forty-one of hers."""

    def _month(self):
        pattern = "AAAAAAAAAAAA" "UAUA" "AAAA" "UA" "AAAAAAAAA" "UA" "AAAAAAA" \
                  "UAUAUA" "AA" "UA" "AAAAAAA" "UA"
        turns, k, n = [], 0, 0
        for c in pattern:
            n += 1
            turns.append(_t("user" if c == "U" else "assistant", f"{c}{n}", k))
            k += 1
        return turns

    def test_every_one_of_his_turns_is_kept(self):
        month = self._month()
        buf = []
        for t in month:
            buf = trim_to_cap(buf + [t], 30)

        assert _his(buf) == _his(month)

    def test_a_long_silence_costs_him_nothing(self):
        buf = []
        for t in self._month():
            buf = trim_to_cap(buf + [t], 30)
        before = _his(buf)

        for i in range(180):   # sixty days, three openers a day
            buf = trim_to_cap(buf + [_t("assistant", f"late {i}", 1000 + i)], 30)

        assert _his(buf) == before


class TestBothWritePathsUseIt:
    @pytest.mark.asyncio
    async def test_the_background_append_path(self, tmp_path):
        """Proactive writes here, while another recipient may be active."""
        mem = ShortTermMemory(max_turns=4, data_dir=tmp_path)
        await mem.append_for_recipient("feishu:x", "user", "周末在家拼乐高")
        for i in range(6):
            await mem.append_for_recipient("feishu:x", "assistant", f"opener {i}")

        kept = await mem.snapshot_for_recipient("feishu:x")

        assert "周末在家拼乐高" in _his(kept) and len(kept) == 4

    @pytest.mark.asyncio
    async def test_the_active_buffer(self, tmp_path):
        mem = ShortTermMemory(max_turns=4, data_dir=tmp_path)
        await mem.switch_recipient("feishu:x")
        mem.add_turn("user", "周末在家拼乐高")
        for i in range(6):
            mem.add_turn("assistant", f"opener {i}")

        kept = mem.get_history()

        assert "周末在家拼乐高" in _his(kept) and len(kept) == 4

    @pytest.mark.asyncio
    async def test_a_file_already_over_the_cap_is_trimmed_on_load(self, tmp_path):
        big = ShortTermMemory(max_turns=40, data_dir=tmp_path)
        await big.switch_recipient("feishu:x")
        big.add_turn("user", "周末在家拼乐高")
        for i in range(10):
            big.add_turn("assistant", f"opener {i}")
        await big.persist_current()

        small = ShortTermMemory(max_turns=4, data_dir=tmp_path)
        await small.switch_recipient("feishu:x")

        assert len(small.get_history()) == 4
        assert "周末在家拼乐高" in _his(small.get_history())
