"""How far back the anti-repeat guard can see, versus how much the prompt shows.

One cap of 10 served both. That was weeks of history while the wait doubled
per unanswered message; with the doubling removed it is 4 messages a day and
so 2.5 days — the guard's memory collapses exactly when it becomes the only
thing limiting her, since the timer no longer is.

The prompt block stays small for its own reason: it is context, not an
archive.
"""

from lingxi.temporal.proactive import ProactiveScheduler


def _sched():
    s = object.__new__(ProactiveScheduler)
    s._recent_proactive = {}
    s._max_recent_proactive = ProactiveScheduler._max_recent_proactive
    s._max_dedup_history = ProactiveScheduler._max_dedup_history
    s._history_path = None
    return s


def test_the_guard_remembers_more_than_the_prompt_shows():
    assert (ProactiveScheduler._max_dedup_history
            > ProactiveScheduler._max_recent_proactive)


def test_the_guard_covers_more_than_a_week_at_four_a_day():
    """Flat 6h re-engage is at most four messages a day when ignored."""
    assert ProactiveScheduler._max_dedup_history / 4 >= 7


def test_stored_history_is_trimmed_to_the_dedup_window():
    s = _sched()
    key = "feishu:oc_x"
    for i in range(ProactiveScheduler._max_dedup_history + 15):
        s._remember_proactive(key, f"第{i}条")

    kept = s._recent_proactive[key]
    assert len(kept) == ProactiveScheduler._max_dedup_history
    assert kept[-1]["text"] == f"第{ProactiveScheduler._max_dedup_history + 14}条"


def test_a_message_from_beyond_the_prompt_block_is_still_compared():
    """The repeat this exists for: said 20 messages ago, out of prompt view."""
    s = _sched()
    key = "feishu:oc_x"
    s._remember_proactive(key, "你那个测试还测不测啦")
    for i in range(20):
        s._remember_proactive(key, f"无关的第{i}条")

    previous = [e["text"] for e in s._recent_proactive[key]]

    assert "你那个测试还测不测啦" in previous
    assert len(previous) > ProactiveScheduler._max_recent_proactive
