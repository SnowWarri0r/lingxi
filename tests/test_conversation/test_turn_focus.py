"""Tests for turn_focus — last-question detection + reminder rendering.

CC-pattern context refactor. The reminder uses the user-message channel
for high-recency placement of dynamic state, since system prompt loses
that signal under the weight of stable persona/rule blocks.

build_turn_focus_reminder lives on PromptBuilder (it needs persona-aware
rendering). detect_last_assistant_question is a pure utility here.
"""

from datetime import datetime

from lingxi.conversation.turn_focus import detect_last_assistant_question
from lingxi.memory.short_term import ConversationTurn
from lingxi.persona.models import Identity, PersonaConfig
from lingxi.persona.prompt_builder import PromptBuilder


def _builder():
    return PromptBuilder(
        PersonaConfig(name="T", identity=Identity(full_name="T"))
    )


def build_turn_focus_reminder(last_assistant_question=None):
    """Compatibility shim — calls PromptBuilder method with question only."""
    return _builder().build_turn_focus_reminder(
        last_assistant_question=last_assistant_question,
    )


def _t(role: str, content: str) -> ConversationTurn:
    return ConversationTurn(role=role, content=content, timestamp=datetime.now())


class TestDetection:
    def test_empty_history_returns_none(self):
        assert detect_last_assistant_question([]) is None

    def test_no_assistant_returns_none(self):
        history = [_t("user", "hello"), _t("user", "still hello")]
        assert detect_last_assistant_question(history) is None

    def test_assistant_without_question_returns_none(self):
        history = [
            _t("assistant", "嗯 我也觉得"),
            _t("user", "确实"),
        ]
        assert detect_last_assistant_question(history) is None

    def test_question_mark_detected(self):
        history = [
            _t("assistant", "你今天回家了吗?"),
            _t("user", "还不在呢"),
        ]
        result = detect_last_assistant_question(history)
        assert result == "你今天回家了吗?"

    def test_chinese_question_mark_detected(self):
        history = [
            _t("assistant", "几点下班？"),
            _t("user", "8点"),
        ]
        assert detect_last_assistant_question(history) == "几点下班？"

    def test_吗_at_end_detected(self):
        history = [
            _t("assistant", "你现在在家了吗"),
            _t("user", "还不在呢"),
        ]
        assert detect_last_assistant_question(history) == "你现在在家了吗"

    def test_呢_at_end_detected(self):
        history = [
            _t("assistant", "今天怎么样呢"),
            _t("user", "还行"),
        ]
        assert detect_last_assistant_question(history) == "今天怎么样呢"

    def test_interrogative_in_tail_detected(self):
        history = [
            _t("assistant", "你今天吃了啥"),
            _t("user", "面"),
        ]
        result = detect_last_assistant_question(history)
        assert result == "你今天吃了啥"

    def test_skips_trailing_user_messages(self):
        # User has sent 2 follow-ups since Aria's question
        history = [
            _t("assistant", "你回家了吗"),
            _t("user", "还不在"),
            _t("user", "在加班"),  # current message
        ]
        # Should still find the assistant's question
        assert detect_last_assistant_question(history) == "你回家了吗"

    def test_multi_bubble_picks_first_question(self):
        history = [
            _t("assistant", "嗯\n\n你回家了吗\n\n好困啊"),
            _t("user", "还不在"),
        ]
        # First question bubble surfaces
        assert detect_last_assistant_question(history) == "你回家了吗"

    def test_multi_bubble_no_question_returns_none(self):
        history = [
            _t("assistant", "嗯\n\n我也是\n\n反正会过去的"),
            _t("user", "对"),
        ]
        assert detect_last_assistant_question(history) is None

    def test_long_question_truncated(self):
        # Cap at 200 chars to keep reminder lean
        long_q = "你" + "今天怎么样" * 50 + "?"  # > 200 chars
        history = [_t("assistant", long_q), _t("user", "还行")]
        result = detect_last_assistant_question(history)
        assert result is not None
        assert len(result) <= 200


class TestReminderBuilding:
    def test_no_question_returns_none(self):
        # Skip rendering when there's nothing dynamic to surface
        assert build_turn_focus_reminder() is None
        assert build_turn_focus_reminder(last_assistant_question=None) is None
        assert build_turn_focus_reminder(last_assistant_question="") is None

    def test_question_renders_as_system_reminder(self):
        result = build_turn_focus_reminder(last_assistant_question="你回家了吗")
        assert result is not None
        assert result.startswith("<system-reminder>")
        assert result.rstrip().endswith("</system-reminder>")
        assert "你回家了吗" in result

    def test_reminder_includes_anti_pattern_warnings(self):
        result = build_turn_focus_reminder(last_assistant_question="你回家了吗")
        # Critical anti-patterns explicitly called out
        assert "胶水" in result          # 不要用'对'起头当胶水
        assert "通用劝慰" in result      # 不要切到通用劝慰
        assert "旧话题" in result        # 不要跳回旧话题

    def test_reminder_includes_concrete_good_examples(self):
        result = build_turn_focus_reminder(last_assistant_question="你回家了吗")
        # Concrete good responses given as guidance
        assert "加班" in result or "几点" in result

    def test_caveat_at_end(self):
        result = build_turn_focus_reminder(last_assistant_question="你吃饭了吗")
        # Caveat reminds model that user's actual message is the task
        assert "对方真正的话" in result or "状态提醒" in result
