"""Openers keep track of which side of her life they came from.

Over 44 openers, 12 were practice and 3 were news from the scene he follows,
and runs of the same subject went out back to back. Replayed as a chain over
a real week of opener prompts: news 2 of 22 → 7 of 29, practice and writing
45% → 34%, same subject as the one before 4 → 2.
"""

from types import SimpleNamespace

from lingxi.temporal.proactive import (
    OPENER_ASPECTS,
    _aspect_line,
    build_opener_news_block,
    parse_proactive_output,
)


def test_the_aspect_she_names_is_kept():
    out = parse_proactive_output(
        '今天新单上线了！\n===META===\n{"should_send": true, "aspect": "圈里新鲜事"}')
    assert out["aspect"] == "圈里新鲜事"


def test_an_aspect_off_the_list_is_dropped():
    out = parse_proactive_output(
        '今天新单上线了！\n===META===\n{"should_send": true, "aspect": "随便"}')
    assert out["aspect"] == ""


def test_the_line_names_what_the_last_three_were_about():
    recent = [{"aspect": "天气"}, {"aspect": "练习排练"},
              {"aspect": "练习排练"}, {"aspect": "练习排练"}]
    line = _aspect_line(recent)
    assert "练习排练×3" in line and "天气" not in line
    assert "另一个方面" in line


def test_no_line_before_any_aspect_is_known():
    assert _aspect_line([{"aspect": ""}, {"text": "旧的"}]) == ""


def test_every_scan_item_is_offered_as_an_opener():
    facts = [SimpleNamespace(content=f"消息{i}") for i in range(7)]
    block = build_opener_news_block(facts)
    assert "拿来起头" in block
    assert block.count("\n- ") + block.startswith("- ") == 5


def test_no_scan_no_block():
    assert build_opener_news_block([]) == ""


def test_a_copied_code_fence_does_not_go_out():
    out = parse_proactive_output(
        '```\n诶 今天排练好累\n===META===\n{"should_send": true}\n```')
    assert out["message"] == "诶 今天排练好累"


def test_the_list_has_a_place_for_news():
    assert "圈里新鲜事" in OPENER_ASPECTS
