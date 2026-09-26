"""When he comes back, she should know how many times she called.

Nothing accumulated. The only silence signal was three prompt lines keyed off
`now - last_interaction`, recomputed each turn — ignore her eight days and she
gets one sentence; reply once and the gap is zero again, as if it had not
happened. The count that does accumulate, consecutive_proactive_count, was
never read by the chat prompt, and record_interaction zeroed it before the
prompt was built anyway.

The line states what happened and what she is like, not what to say: scripting
her sentence is what makes a persona read as a form letter.
"""

from datetime import datetime, timedelta

from lingxi.persona.prompt_builder import build_unanswered_block


NOW = datetime(2026, 9, 18, 17, 0)


def test_no_unanswered_messages_means_no_block():
    assert build_unanswered_block(0, NOW - timedelta(hours=3), NOW) is None


def test_a_single_unanswered_message_is_not_worth_mentioning():
    """Everyone misses one. This is for a pattern, not an instance."""
    assert build_unanswered_block(1, NOW - timedelta(days=1), NOW) is None


def test_a_few_unanswered_messages_carry_the_count():
    block = build_unanswered_block(3, NOW - timedelta(days=2), NOW)

    assert block is not None
    assert "3" in block


def test_many_unanswered_messages_still_carry_the_count():
    block = build_unanswered_block(7, NOW - timedelta(days=8), NOW)

    assert "7" in block


class TestOneNightIsNotBeingIgnored:
    """She sends three a day, so the count crosses 2 every single night.

    Measured after the exponential re-engage wait came out: a steady 3 sends
    per day, and his replies came back with 2, 3 and 3 unanswered after gaps
    of 20h, 23.7h and 25h — ordinary evenings. The one that was worth saying
    sat at 6 unanswered across 45 hours. A count alone cannot tell those
    apart; the stretch can.
    """

    def test_an_ordinary_overnight_gap_says_nothing(self):
        assert build_unanswered_block(3, NOW - timedelta(hours=23.7), NOW) is None

    def test_a_long_workday_gap_says_nothing(self):
        assert build_unanswered_block(2, NOW - timedelta(hours=20), NOW) is None

    def test_a_day_and_a_bit_still_says_nothing(self):
        assert build_unanswered_block(3, NOW - timedelta(hours=25), NOW) is None

    def test_the_stretch_that_was_worth_saying_still_speaks(self):
        """The real 09-20 turn: 6 unanswered across 45 hours."""
        block = build_unanswered_block(6, NOW - timedelta(hours=45), NOW)

        assert block is not None and "6" in block

    def test_one_unanswered_message_stays_silent_however_long(self):
        assert build_unanswered_block(1, NOW - timedelta(days=9), NOW) is None


def test_it_says_how_long_the_stretch_was():
    block = build_unanswered_block(5, NOW - timedelta(days=8), NOW)

    assert "8天" in block


def test_it_tells_her_to_drop_it_afterwards():
    """Chosen behaviour: say it, then it is over — no grudge carried."""
    block = build_unanswered_block(5, NOW - timedelta(days=8), NOW)

    assert "过去" in block or "翻篇" in block


def test_it_describes_her_disposition_rather_than_supplying_a_line():
    """A scripted sentence is the thing that reads as a form letter."""
    block = build_unanswered_block(5, NOW - timedelta(days=8), NOW)

    assert "「" not in block and "『" not in block


def test_a_bigger_stretch_is_not_a_softer_one():
    """Whatever the wording, more unanswered must not read as less."""
    few = build_unanswered_block(2, NOW - timedelta(days=1), NOW)
    many = build_unanswered_block(8, NOW - timedelta(days=10), NOW)

    assert few != many


def test_a_missing_timestamp_still_produces_the_count():
    block = build_unanswered_block(4, None, NOW)

    assert block is not None and "4" in block
