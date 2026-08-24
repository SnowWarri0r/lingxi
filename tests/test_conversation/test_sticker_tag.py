"""The trailing `#表情 X` line: how the single-pass responder asks for a sticker."""

from lingxi.conversation.output_schema import parse_turn_output


def test_tag_is_parsed_and_kept_out_of_the_speech():
    """The marker must never reach the chat window."""
    out = parse_turn_output("哈哈哈别慌！到时候我先说\n#表情 害羞")
    assert out.sticker == "害羞"
    assert "#表情" not in out.speech
    assert out.speech.strip() == "哈哈哈别慌！到时候我先说"


def test_full_width_colon_and_spacing_are_tolerated():
    for raw in ("在的\n#表情 开心", "在的\n#表情：开心", "在的\n# 表情: 开心",
                "在的\n  #表情  开心  "):
        out = parse_turn_output(raw)
        assert out.sticker == "开心", raw
        assert "表情" not in out.speech, raw


def test_no_tag_means_no_sticker():
    out = parse_turn_output("知道啊 你是程序员 怎么了")
    assert out.sticker == ""
    assert out.speech.strip() == "知道啊 你是程序员 怎么了"


def test_a_hash_mid_sentence_is_not_a_tag():
    """Only a line that is nothing but the tag counts, so ordinary text
    mentioning 表情 cannot silently trigger an image."""
    out = parse_turn_output("我发的那个#表情 你看到没")
    assert out.sticker == ""
    assert "表情" in out.speech


def test_last_tag_wins_when_several_are_written():
    out = parse_turn_output("一句\n#表情 开心\n二句\n#表情 害羞")
    assert out.sticker == "害羞"
    assert "#表情" not in out.speech


def test_meta_json_still_works_for_the_claude_path():
    """Claude-with-tools emits the JSON block and still does; both are read."""
    out = parse_turn_output('难受死了吧 别硬撑\n===META===\n{"mood":"心疼","sticker":"心疼"}')
    assert out.sticker == "心疼"
    assert out.mood_label == "心疼"


def test_json_sticker_beats_a_tag_when_both_appear():
    out = parse_turn_output('在的\n#表情 开心\n===META===\n{"sticker":"心疼"}')
    assert out.sticker == "心疼"
