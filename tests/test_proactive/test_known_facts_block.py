"""The proactive prompt must carry consolidated facts, not just raw fragments."""

from datetime import datetime

from lingxi.facts.models import Fact, FactType, Source
from lingxi.temporal.proactive import _format_known_block


def _fact(content, day=22):
    return Fact(subject="user:feishu:x", content=content, source=Source.USER_STATED,
                type=FactType.PATTERN, ts=datetime(2026, 8, day, 15, 0), importance=6)


def test_facts_are_rendered_with_their_dates():
    out = _format_known_block([_fact("对方8月22日去成都参加Liyuu活动")])
    assert "08-22" in out
    assert "对方8月22日去成都参加Liyuu活动" in out


def test_the_block_says_it_outranks_the_raw_fragments():
    """The fragments above it are truncated originals; these are cleaned up.

    Without saying which wins, she reconstructed a trip from cut-off chat
    lines and produced 「连着两天跑漫展」 for a one-day Liyuu event, borrowing
    the 漫展 from a separate future plan.
    """
    out = _format_known_block([_fact("x")])
    assert "以这里为准" in out


def test_it_tells_her_not_to_infer_the_parts_that_are_missing():
    out = _format_known_block([_fact("x")])
    assert "别自己推" in out


def test_no_facts_renders_nothing():
    assert _format_known_block([]) == ""


def test_several_facts_all_appear():
    out = _format_known_block([
        _fact("对方8月22日去成都参加Liyuu活动"),
        _fact("对方说国庆会去广州漫展见Liyuu", day=24),
    ])
    assert "成都" in out and "广州漫展" in out
    assert out.count("- [") == 2


class _Turn:
    def __init__(self, content, minute=0):
        from datetime import datetime
        self.content = content
        self.timestamp = datetime(2026, 8, 24, 15, minute)


def test_long_messages_keep_the_tail_that_disambiguates():
    """80 chars cut his longest line at 「说了自己从24年广州亚」.

    The tail is where it says which event and whose — losing it is how the
    Chengdu trip and a future Guangzhou con became one thing.
    """
    from lingxi.temporal.proactive import _format_user_recent

    long = ("是鲤鱼抽到的我，上台的时候我还给她展示了一下，还夸了她晚场的衣服比下午场的更可爱了，"
            "还说了国庆会去广州漫展见她，手写信也给她了，说了自己从24年广州亚巡开始跑现地")
    out = _format_user_recent([_Turn(long)])
    assert "国庆会去广州漫展见她" in out
    assert "24年广州亚巡开始跑现地" in out


def test_a_repeated_message_does_not_take_two_slots():
    from lingxi.temporal.proactive import _format_user_recent

    dup = "是鲤鱼抽到的我，哈哈"
    out = _format_user_recent([_Turn(dup, 33), _Turn(dup, 34), _Turn("另一句", 35)])
    assert out.count(dup) == 1
    assert "另一句" in out


def test_something_absurdly_long_is_still_bounded():
    from lingxi.temporal.proactive import _format_user_recent

    out = _format_user_recent([_Turn("啊" * 500)])
    assert out.endswith("…")
    assert len(out) < 300


def test_empty_messages_are_dropped():
    from lingxi.temporal.proactive import _format_user_recent

    assert _format_user_recent([_Turn("   ")]) == ""
