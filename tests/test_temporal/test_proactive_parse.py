"""An opener that omits ===META=== is still an opener.

The parse treated a missing delimiter as "the model spat raw JSON (legacy)"
and put the whole reply in meta_part, leaving speech empty — so should_send
came out False and the message was dropped without a log line. Measured over
119 logged openers: 66 had no META block, 64 of them carried a real message.
More than half of everything she wrote was discarded on a format technicality,
and the scheduler simply retried five minutes later.

Lifting the sticker tag is part of the same fix: once these messages send, an
unlifted `#表情 开心` line would go out as visible text.
"""

from lingxi.temporal.proactive import parse_proactive_output


class TestNoMetaBlock:
    def test_a_bare_message_is_the_message(self):
        out = parse_proactive_output("今天下午那行词还是没抓住 千砂都说我最近想太多了")

        assert out["should_send"] is True
        assert out["message"] == "今天下午那行词还是没抓住 千砂都说我最近想太多了"

    def test_a_multi_line_bare_message_survives_whole(self):
        raw = "今天彩排站在台上往下看\n\n突然有点想哭"
        out = parse_proactive_output(raw)

        assert out["message"] == raw
        assert out["should_send"] is True

    def test_an_empty_reply_sends_nothing(self):
        assert parse_proactive_output("   ")["should_send"] is False


class TestStickerTag:
    def test_the_tag_never_goes_out_as_text(self):
        out = parse_proactive_output("场刊翻到我们那页了！拍给你看\n#表情 开心")

        assert "#表情" not in out["message"]
        assert out["message"] == "场刊翻到我们那页了！拍给你看"

    def test_the_intent_is_carried_out_for_the_sender(self):
        out = parse_proactive_output("场刊翻到我们那页了\n#表情 开心")

        assert out["sticker"] == "开心"

    def test_a_tag_with_meta_present_is_also_lifted(self):
        out = parse_proactive_output(
            '想你了\n#表情 撒娇\n===META===\n{"should_send": true, "inner": "x"}')

        assert out["message"] == "想你了"
        assert out["sticker"] == "撒娇"
        assert out["should_send"] is True

    def test_no_tag_means_no_sticker(self):
        assert parse_proactive_output("就说一句")["sticker"] == ""

    def test_a_tag_only_reply_sends_nothing(self):
        """Nothing to say plus a mood is not a message."""
        out = parse_proactive_output("#表情 开心")

        assert out["should_send"] is False


class TestMetaStillWins:
    def test_an_explicit_decline_is_honoured(self):
        out = parse_proactive_output(
            '\n===META===\n{"should_send": false, "inner": "没什么想说的"}')

        assert out["should_send"] is False

    def test_a_decline_with_text_is_still_a_decline(self):
        out = parse_proactive_output(
            '算了\n===META===\n{"should_send": false, "inner": "时间不合适"}')

        assert out["should_send"] is False

    def test_the_string_false_is_not_truthy(self):
        out = parse_proactive_output(
            '话\n===META===\n{"should_send": "false"}')

        assert out["should_send"] is False

    def test_speech_and_meta_both_land(self):
        out = parse_proactive_output(
            '刚练完 嗓子有点哑\n===META===\n{"should_send": true, "inner": "想说"}')

        assert out["message"] == "刚练完 嗓子有点哑"
        assert out["should_send"] is True

    def test_a_bare_json_reply_is_still_read_as_meta(self):
        """The legacy shape the old fallback was written for."""
        out = parse_proactive_output(
            '{"should_send": true, "message": "早呀", "inner": "打个招呼"}')

        assert out["should_send"] is True
        assert "should_send" not in out["message"]
