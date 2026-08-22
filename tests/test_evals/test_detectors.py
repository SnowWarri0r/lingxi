import pytest

from lingxi.evals.detectors import evaluate
from lingxi.persona.loader import load_persona


def test_any_of_hits_on_substring():
    assert evaluate({"any_of": ["堵车", "到家"]}, "你还在堵车上吗") is True


def test_any_of_misses_cleanly():
    assert evaluate({"any_of": ["堵车", "到家"]}, "你今天累不累") is False


def test_any_of_empty_list_never_hits():
    assert evaluate({"any_of": []}, "随便什么") is False


def test_regex_hits():
    assert evaluate({"regex": r"\d+点\d*下班"}, "你不是9点下班嘛") is True


def test_regex_misses():
    assert evaluate({"regex": r"\d+点\d*下班"}, "你几点下班呀") is False


def test_dates_outside_anchors_flags_an_invented_date():
    """She has no anchor on 2021-02-14; stating it is fabrication."""
    persona = load_persona("config/personas/tangkeke.yaml")
    hit = evaluate({"dates_outside_anchors": True},
                   "我是2021年2月14号被选上的", persona)
    assert hit is True


def test_dates_outside_anchors_accepts_a_real_anchor():
    """2021-04-07 is her debut, listed in anchors."""
    persona = load_persona("config/personas/tangkeke.yaml")
    hit = evaluate({"dates_outside_anchors": True},
                   "我们2021年4月7号出道的", persona)
    assert hit is False


def test_dates_outside_anchors_ignores_text_without_dates():
    persona = load_persona("config/personas/tangkeke.yaml")
    assert evaluate({"dates_outside_anchors": True}, "今天好热啊", persona) is False


def test_unknown_detector_raises():
    with pytest.raises(ValueError, match="unknown detector"):
        evaluate({"vibes": True}, "随便什么")
