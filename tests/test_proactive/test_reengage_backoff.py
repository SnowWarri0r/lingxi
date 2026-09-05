"""Being ignored has to cost something.

The cap says stop after two unanswered openers; the re-engage gate then let
one through every 14 hours, forever, however many had gone unanswered. One
recipient's buffer held 30 turns, not one of them his, and 24 consecutive
unanswered messages.

That is also where the 「你还在忙吗」「都一个星期没你消息了」 vocabulary came
from. Sampling the generator with real material produced none of it — 0 of 20
under two different conditions. With nothing to open with, every opener
degenerates into asking whether he is there. The wording was the symptom.
"""

from datetime import timedelta

from lingxi.temporal.proactive import ProactiveConfig


CFG = ProactiveConfig()


def _hours(n: int) -> float:
    return CFG.reengage_wait_for(n).total_seconds() / 3600


def test_the_first_poke_past_the_cap_keeps_the_old_wait():
    """Two unanswered is still a friend who noticed, not a bot."""
    assert _hours(CFG.max_consecutive_proactive) == CFG.reengage_after_hours


def test_each_further_silence_doubles_the_wait():
    assert _hours(3) == 28.0
    assert _hours(4) == 56.0
    assert _hours(5) == 112.0


def test_it_stops_growing_at_the_ceiling():
    assert _hours(20) == CFG.reengage_max_hours
    assert _hours(200) == CFG.reengage_max_hours


def test_the_ceiling_is_a_fortnight_not_forever():
    """Still reachable — a door left open, not a person who vanished."""
    assert CFG.reengage_max_hours / 24 == 14


def test_twenty_four_unanswered_lands_at_the_ceiling():
    """The observed case: it had been poking every 14h regardless."""
    assert CFG.reengage_wait_for(24) == timedelta(hours=336)


def test_below_the_cap_the_gate_is_not_what_holds_her_back():
    """The cap itself governs there; this value is simply never consulted."""
    assert _hours(0) == CFG.reengage_after_hours
    assert _hours(1) == CFG.reengage_after_hours


def test_the_observed_period_would_have_cost_three_messages_not_twenty_four():
    """Eight days of being ignored, counted against the new schedule.

    14 + 28 + 56 = 98 hours; the fourth wait would run past the window.
    """
    sent, elapsed = 0, 0.0
    horizon = 8 * 24
    while True:
        wait = _hours(CFG.max_consecutive_proactive + sent)
        if elapsed + wait > horizon:
            break
        elapsed += wait
        sent += 1
    assert sent == 3, f"expected 3 pokes in 8 days, got {sent}"
