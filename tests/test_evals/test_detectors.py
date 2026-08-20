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
