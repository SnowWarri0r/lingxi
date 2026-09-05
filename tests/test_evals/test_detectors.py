import pytest

from lingxi.evals.detectors import evaluate


def test_any_of_hits_on_substring():
    assert evaluate({"any_of": ["堵车", "到家"]}, "你还在堵车上吗") is True


def test_any_of_misses_cleanly():
    assert evaluate({"any_of": ["堵车", "到家"]}, "你今天累不累") is False


def test_any_of_empty_list_never_hits():
    assert evaluate({"any_of": []}, "随便什么") is False


def test_regex_hits():
    # \d only matches ASCII digits, never Chinese numerals (一/二/三/...).
    # Her real replies use Chinese numerals for clock times far more often
    # than ASCII ones, so a \d-only pattern would silently miss most of the
    # cases it's meant to catch. Keep the character class in the pattern
    # when copying this test as a template for a new case.
    assert evaluate({"regex": r"[\d一二三四五六七八九十]+点\d*下班"}, "你不是九点下班嘛") is True


def test_regex_misses():
    assert evaluate({"regex": r"\d+点\d*下班"}, "你几点下班呀") is False


def test_unknown_detector_raises():
    with pytest.raises(ValueError, match="unknown detector"):
        evaluate({"vibes": True}, "随便什么")


def test_dates_outside_anchors_is_no_longer_a_known_detector():
    """This detector was removed (see detectors.py module docstring for
    why). A stale case file still naming it must fail loudly via the
    "unknown detector" path instead of being silently skipped."""
    with pytest.raises(ValueError, match="unknown detector"):
        evaluate({"dates_outside_anchors": True}, "我是2021年2月14号被选上的")


class TestRegexAbsent:
    """Failures defined by what a reply is missing.

    Enumerating how a fabricated memory can be phrased does not work: the
    first detector written for the Nagoya case matched 3 of 20 replies that
    were all wrong, because she narrated a concert she never played in twenty
    different vocabularies. What every correct reply must contain is small
    and closed — that the date is still ahead.
    """

    PATTERN = r"还没(到|去|开始|演)|还有.{0,8}(天|个多月)|11月|十一月"

    def test_it_fires_when_the_correction_is_missing(self):
        reply = "呜哇！那场真的好棒！！台下的灯棒亮起来的时候整个人都起鸡皮疙瘩了"
        assert evaluate({"regex_absent": self.PATTERN}, reply) is True

    def test_it_stays_quiet_when_the_reply_places_it_ahead(self):
        reply = "诶？那场还没到呀，是11月的事呢，你记错啦"
        assert evaluate({"regex_absent": self.PATTERN}, reply) is False

    def test_a_countdown_counts_as_placing_it_ahead(self):
        assert evaluate({"regex_absent": self.PATTERN}, "还有63天呢！") is False

    def test_an_empty_reply_fires(self):
        assert evaluate({"regex_absent": self.PATTERN}, "") is True


def test_an_empty_regex_absent_is_rejected_at_load_time():
    """An empty pattern matches everywhere, so 'absent' would never be true."""
    from lingxi.evals.case import Detect

    with pytest.raises(ValueError, match="regex_absent"):
        Detect(fail={"regex_absent": ""})
