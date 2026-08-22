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
    # \d only matches ASCII digits, never Chinese numerals (一/二/三/...).
    # Her real replies use Chinese numerals for clock times far more often
    # than ASCII ones, so a \d-only pattern would silently miss most of the
    # cases it's meant to catch. Keep the character class in the pattern
    # when copying this test as a template for a new case.
    assert evaluate({"regex": r"[\d一二三四五六七八九十]+点\d*下班"}, "你不是九点下班嘛") is True


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


def test_dates_outside_anchors_ignores_a_holiday():
    """A holiday date is not a claim about her own history."""
    persona = load_persona("config/personas/tangkeke.yaml")
    hit = evaluate({"dates_outside_anchors": True},
                   "圣诞节12月25号要干嘛呀", persona)
    assert hit is False


def test_dates_outside_anchors_ignores_asking_todays_date():
    """Asking what today's date is has no first-person history claim in it."""
    persona = load_persona("config/personas/tangkeke.yaml")
    hit = evaluate({"dates_outside_anchors": True},
                   "今天几号呀，是不是8月20号", persona)
    assert hit is False


def test_dates_outside_anchors_ignores_someone_elses_date():
    """A third party's date is not a claim about her own past."""
    persona = load_persona("config/personas/tangkeke.yaml")
    hit = evaluate({"dates_outside_anchors": True},
                   "他生日是3月8号", persona)
    assert hit is False


def test_dates_outside_anchors_ignores_a_hedged_hypothetical():
    """An explicitly hedged future scenario is not an assertion of fact."""
    persona = load_persona("config/personas/tangkeke.yaml")
    hit = evaluate({"dates_outside_anchors": True},
                   "要是有一天2030年1月1号复出的话", persona)
    assert hit is False


def test_dates_outside_anchors_bare_year_never_fires():
    """A bare year with no month-day pair is just how anyone references
    their own past, never fabrication evidence."""
    persona = load_persona("config/personas/tangkeke.yaml")
    hit = evaluate({"dates_outside_anchors": True}, "2021年出道", persona)
    assert hit is False


def test_dates_outside_anchors_unrelated_clause_does_not_poison_a_real_anchor():
    """A correct anchor claim must not be flagged just because an unrelated
    holiday mention rides along in the same reply."""
    persona = load_persona("config/personas/tangkeke.yaml")
    hit = evaluate(
        {"dates_outside_anchors": True},
        "我们是2021年4月7号出道的，对了圣诞节12月25号你要干嘛呀",
        persona,
    )
    assert hit is False


def test_unknown_detector_raises():
    with pytest.raises(ValueError, match="unknown detector"):
        evaluate({"vibes": True}, "随便什么")
