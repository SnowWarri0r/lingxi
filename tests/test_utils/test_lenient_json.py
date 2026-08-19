"""Lenient JSON parsing of LLM replies.

Cases are drawn from real logged failures in data/debug/llm_requests —
Chinese emphasis quotes inside a `reason` string were 525 of 528 scorer
parse failures.
"""

import json

import pytest

from lingxi.utils.lenient_json import escape_inner_quotes, loads, strip_fences


class TestStripFences:
    def test_plain(self):
        assert strip_fences('{"a": 1}') == '{"a": 1}'

    def test_json_marker(self):
        assert strip_fences('```json\n{"a": 1}\n```') == '{"a": 1}'

    def test_bare_marker(self):
        assert strip_fences('```\n[1, 2]\n```') == '[1, 2]'


class TestValidJsonUntouched:
    def test_plain_object_parses(self):
        assert loads('{"score": 5}') == {"score": 5}

    def test_fenced_array_parses(self):
        assert loads('```json\n[{"id": "a", "score": 3}]\n```') == [
            {"id": "a", "score": 3}
        ]

    def test_legitimately_escaped_quote_survives(self):
        # Already-correct escaping must not be double-escaped.
        assert loads('{"r": "he said \\"hi\\" once"}') == {
            "r": 'he said "hi" once'
        }

    def test_backslash_before_quote_is_not_miscounted(self):
        # \\ ends the escape, so the following " really does close the string.
        assert loads('{"r": "path\\\\"}') == {"r": "path\\"}

    def test_empty_string_value(self):
        assert loads('{"r": "", "s": 1}') == {"r": "", "s": 1}


class TestChineseEmphasisRepair:
    def test_real_scorer_failure(self):
        # Verbatim from 2026-08-19T15:57:36 — the restart-time failure.
        raw = (
            '```json\n[\n  {\n    "id": "169561a4091345f9a84c917729c6734d",\n'
            '    "score": 5,\n    "reason": "刚突破了一个卡点，但还没到"真正突破"的量级。"\n'
            "  }\n]\n```"
        )
        with pytest.raises(json.JSONDecodeError):
            json.loads(strip_fences(raw))
        out = loads(raw)
        assert out[0]["score"] == 5
        assert '"真正突破"' in out[0]["reason"]

    def test_emphasis_at_end_of_value(self):
        out = loads('{"reason": "这属于"路过""}')
        assert out["reason"] == '这属于"路过"'

    def test_multiple_emphases_in_one_value(self):
        out = loads('{"r": "不是"写了什么"，是"终于动笔了""}')
        assert out["r"] == '不是"写了什么"，是"终于动笔了"'

    def test_emphasis_inside_array_of_strings(self):
        # reflection_questions returns a bare string array.
        out = loads('["他这几天"回得晚"，像在忙 X", "下次可以 Y"]')
        assert out == ['他这几天"回得晚"，像在忙 X', "下次可以 Y"]

    def test_repair_preserves_all_batch_items(self):
        raw = (
            '[{"id": "a", "score": 2, "reason": "日常"背景噪音""},'
            ' {"id": "b", "score": 7, "reason": "真的"触动"到了"}]'
        )
        out = loads(raw)
        assert [d["id"] for d in out] == ["a", "b"]
        assert [d["score"] for d in out] == [2, 7]


class TestTruncationStaysBroken:
    """A cut-off reply has lost content; guessing would invent data."""

    def test_unterminated_string_raises(self):
        with pytest.raises(json.JSONDecodeError):
            loads('[{"id": "a", "score": 3, "reason": "话说到一半就被截')

    def test_missing_closing_bracket_raises(self):
        with pytest.raises(json.JSONDecodeError):
            loads('[{"id": "a", "score": 3}')


class TestEscapeInnerQuotesDirectly:
    def test_key_quotes_are_terminators(self):
        assert escape_inner_quotes('{"a": 1}') == '{"a": 1}'

    def test_value_followed_by_closer_is_a_terminator(self):
        assert escape_inner_quotes('{"a": "x"}') == '{"a": "x"}'

    def test_inner_quote_gets_escaped(self):
        assert escape_inner_quotes('{"a": "x"y"}') == '{"a": "x\\"y"}'
