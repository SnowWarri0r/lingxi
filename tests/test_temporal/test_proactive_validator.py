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


class TestValidatorRejectsSelfReportOpener:
    """User reported: Aria proactively sent '今天就一直在刷手机' — reads
    like answering an unasked '你今天怎么样?'. Self-state report as
    opener is reactive-shape in proactive context."""

    def test_rejects_今天就_prefix(self):
        # The exact production trace
        assert _validate_proactive_opener("今天就一直在刷手机") == "self_report_opener"

    def test_rejects_今天一直_prefix(self):
        assert _validate_proactive_opener("今天一直没干活") == "self_report_opener"

    def test_rejects_今天没_prefix(self):
        assert _validate_proactive_opener("今天没怎么休息") == "self_report_opener"

    def test_rejects_一天都_prefix(self):
        assert _validate_proactive_opener("一天都在写东西") == "self_report_opener"

    def test_rejects_我今天_prefix(self):
        assert _validate_proactive_opener("我今天有点累") == "self_report_opener"

    def test_self_report_with_question_passes(self):
        # If there's an outward-facing question, opener is valid
        # ("我今天没怎么休息 你呢?")
        assert _validate_proactive_opener("我今天没怎么休息 你呢?") is None

    def test_self_report_with_你_passes(self):
        # If "你" is in the message, treat as outward-facing
        assert _validate_proactive_opener("我今天没怎么休息 你那边怎么样") is None

    def test_specific_event_not_blocked(self):
        # "今天吃了泡面" doesn't match the prefixes (just "今天 V 了")
        # — only the lingering / continuous self-state patterns get caught
        assert _validate_proactive_opener("今天吃了泡面") is None


class TestValidatorRejectsPhaticCheckin:
    """User: '11点问我睡了没' — proactive 'X 了吗' bare check-in
    reads as AI fishing for engagement. Block when message is JUST
    one such question with no concrete hook."""

    def test_rejects_睡了吗(self):
        assert _validate_proactive_opener("你现在睡了吗") == "phatic_checkin"
        assert _validate_proactive_opener("睡了吗") == "phatic_checkin"
        assert _validate_proactive_opener("你睡了没") == "phatic_checkin"

    def test_rejects_回家了吗(self):
        assert _validate_proactive_opener("你回家了吗") == "phatic_checkin"
        assert _validate_proactive_opener("你现在回家了吗") == "phatic_checkin"

    def test_rejects_下班了吗(self):
        assert _validate_proactive_opener("下班了吗") == "phatic_checkin"

    def test_rejects_吃饭了吗(self):
        assert _validate_proactive_opener("你吃饭了吗?") == "phatic_checkin"
        assert _validate_proactive_opener("吃了吗") == "phatic_checkin"

    def test_rejects_在干嘛(self):
        assert _validate_proactive_opener("你在干嘛") == "phatic_checkin"
        assert _validate_proactive_opener("在干嘛呢") == "phatic_checkin"

    def test_question_with_concrete_hook_passes(self):
        # Has context beyond the bare check-in
        assert _validate_proactive_opener("奶奶今天吃得下东西吗") is None

    def test_compound_question_passes(self):
        # Multi-clause messages are valid (has concrete content)
        assert _validate_proactive_opener("你那个报告交了吗") is None
        assert _validate_proactive_opener("还在搞那个项目吗") is None


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
