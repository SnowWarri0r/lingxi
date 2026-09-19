"""The orchestrator is asked what he is doing *now* and was never told when now is.

Item 9 asks for 对方**此刻**人在哪、在干什么, and the prompt carried no clock
at all. On 2026-09-13 the user said he was tired at 15:19 and still up at 19:32; the
orchestrator wrote user_state 「周日下午，在家」 — Sunday afternoon,
written at half past seven in the evening, because it had no way to know.

That string is rendered directly under the real clock as 对方此刻, and marked
as the authority over it. The turn it produced spoke of staying up late, at
19:32.
"""

from datetime import datetime

from lingxi.brain.orchestrator import StateDigest, build_orchestrator_prompt


NOW = datetime(2026, 9, 13, 19, 32)


def _prompt(now=NOW):
    return build_orchestrator_prompt(
        "我还没歇呢",
        StateDigest(activity="在练习室", mood="平静", last_lived=[]),
        {"aria.event": 3},
        now=now,
    )


def test_the_clock_is_in_the_prompt():
    assert "19:32" in _prompt()


def test_the_date_and_weekday_are_there_too():
    prompt = _prompt()

    assert "2026-09-13" in prompt
    assert "周日" in prompt or "星期日" in prompt


def test_the_part_of_day_is_stated_not_left_to_be_inferred():
    assert "晚上" in _prompt()


def test_morning_reads_as_morning():
    assert "早" in _prompt(datetime(2026, 9, 13, 7, 10))


def test_user_state_is_told_to_leave_the_clock_alone():
    """The field is where he is and what he is doing, not what time it is."""
    prompt = _prompt()
    i = prompt.index("user_state")
    item9 = prompt[i:i + 400]

    assert "时段" in item9 or "钟点" in item9


def test_it_still_builds_without_a_clock():
    """Callers that do not pass one must not break."""
    prompt = build_orchestrator_prompt(
        "hi", StateDigest(activity="", mood="", last_lived=[]), {})

    assert "user_state" in prompt
