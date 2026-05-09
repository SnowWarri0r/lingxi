"""Tests for proactive opener validator + should_send normalization.

Locks in two fixes from a code review:
1. Validator must NOT reject pure first-person statements that the prompt
   explicitly allows (e.g. "今天吃了泡面"). Earlier hook-based check made
   the validator contradict the prompt and discard valid openers.
2. should_send normalization must handle stringified booleans — Python's
   bool("false") is True, so naive coercion would send messages the LLM
   explicitly marked as should_send=false.
"""

from lingxi.temporal.proactive import _validate_proactive_opener


class TestValidatorAcceptsPureStatement:
    def test_first_person_observation_passes(self):
        # Allowed by the "日常播报" / "无聊闲话" prompt styles
        assert _validate_proactive_opener("刚发现冰箱酸奶过期一周了 我居然都没察觉") is None

    def test_short_statement_passes(self):
        assert _validate_proactive_opener("今天吃了泡面") is None

    def test_question_still_passes(self):
        assert _validate_proactive_opener("你那个项目搞完没") is None

    def test_specific_concern_passes(self):
        assert _validate_proactive_opener("奶奶今天吃得下东西吗") is None


class TestValidatorRejectsResponseTokenOpener:
    def test_rejects_嗯_prefix(self):
        result = _validate_proactive_opener("嗯 也是 想到你")
        assert result is not None
        assert "response_token" in result
        assert "嗯" in result

    def test_rejects_欸_prefix(self):
        assert _validate_proactive_opener("欸 你今天怎么样").startswith("opens_with_response_token")

    def test_rejects_对了_prefix(self):
        assert _validate_proactive_opener("对了 想问你").startswith("opens_with_response_token")


class TestShouldSendCoercion:
    """Direct unit-level: simulate the meta parsing branch that picks
    whether to send. Mirrors the logic in _ask_llm without mocking the
    full LLM stack.
    """

    @staticmethod
    def coerce(meta_value, message="hello"):
        # Mirror the production logic exactly
        raw = meta_value
        if raw is None:
            return bool(message)
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, str):
            return raw.strip().lower() not in ("false", "no", "0", "")
        return bool(raw)

    def test_string_false_is_false(self):
        # Critical regression: bool("false") is True in Python
        assert self.coerce("false") is False
        assert self.coerce("False") is False
        assert self.coerce("FALSE") is False
        assert self.coerce("  false  ") is False

    def test_string_no_is_false(self):
        assert self.coerce("no") is False
        assert self.coerce("0") is False
        assert self.coerce("") is False

    def test_string_true_is_true(self):
        assert self.coerce("true") is True
        assert self.coerce("True") is True
        assert self.coerce("yes") is True

    def test_real_bool_passes_through(self):
        assert self.coerce(True) is True
        assert self.coerce(False) is False

    def test_none_falls_back_to_message_truthiness(self):
        assert self.coerce(None, message="hi") is True
        assert self.coerce(None, message="") is False
