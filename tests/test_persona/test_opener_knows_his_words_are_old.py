"""An opener only fires after a day or more of silence, so his newest words
are that old — and the prompt told her they were where he is right now.

The time block said 「他刚发的消息说的是他此刻在哪、在干嘛，以那个为准」 and the
block of his messages was headed 「他此刻的状态/在干啥都在这里」, over lines
three days old on 09-24. Right for a reply, where his newest message is the
one he just sent; false on every opener.
"""

from datetime import datetime, timedelta

from lingxi.persona.models import Identity, PersonaConfig
from lingxi.persona.prompt_builder import PromptBuilder

NOW = datetime(2026, 9, 24, 14, 1)


def _time_section(proactive):
    b = PromptBuilder(PersonaConfig(name="唐可可", identity=Identity(full_name="唐可可")))
    return b._build_time_awareness_section(
        NOW, NOW - timedelta(days=3), proactive_mode=proactive)


def test_a_reply_still_ranks_what_he_just_said_above_the_clock():
    assert "以那个为准" in _time_section(proactive=False)


def test_an_opener_does_not_call_three_day_old_words_his_present():
    section = _time_section(proactive=True)

    assert "以那个为准" not in section and "按钟点推" in section


def test_the_opener_heading_over_his_messages_claims_no_present_tense():
    import inspect

    from lingxi.temporal import proactive
    src = inspect.getsource(proactive)

    assert "他此刻的状态/在干啥都在这里" not in src
    assert "对方最后发的几句" in src
