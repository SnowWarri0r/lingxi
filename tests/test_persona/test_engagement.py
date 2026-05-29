"""Tests for derive_engagement_mode + prompt rendering.

Engagement mode is the agency lever — without it, Aria's prompt rules
about "可以敷衍" are soft hints the LLM ignores. With it, withdrawn ==
"engaging fully is the wrong answer", a hard structural switch.

Moved here from test_inner_life when inner_life was retired; engagement
is now derived purely from emotion (the energy<0.3 branch was dropped
along with the numeric inner-state scalars).
"""

from lingxi.persona.engagement import EngagementMode, derive_engagement_mode
from lingxi.persona.models import EmotionState, Identity, PersonaConfig
from lingxi.persona.prompt_builder import PromptBuilder


class TestDerivation:
    def test_no_emotion_returns_full(self):
        assert derive_engagement_mode(None) == EngagementMode.FULL

    def test_heavy_emotion_withdraws(self):
        emotion = EmotionState(dimensions={"悲伤": 0.6})
        assert derive_engagement_mode(emotion) == EngagementMode.WITHDRAWN

    def test_孤独_above_threshold_withdraws(self):
        emotion = EmotionState(dimensions={"孤独": 0.7})
        assert derive_engagement_mode(emotion) == EngagementMode.WITHDRAWN

    def test_heavy_below_threshold_no_withdraw(self):
        emotion = EmotionState(dimensions={"悲伤": 0.3})
        assert derive_engagement_mode(emotion) == EngagementMode.FULL

    def test_provoked_emotion_curt(self):
        emotion = EmotionState(dimensions={"不爽": 0.5})
        assert derive_engagement_mode(emotion) == EngagementMode.CURT

    def test_heavy_overrides_provoked(self):
        emotion = EmotionState(dimensions={"悲伤": 0.6, "不爽": 0.5})
        assert derive_engagement_mode(emotion) == EngagementMode.WITHDRAWN

    def test_normal_emotion_full(self):
        emotion = EmotionState(dimensions={"平静": 0.5, "好奇": 0.3})
        assert derive_engagement_mode(emotion) == EngagementMode.FULL

    def test_flustered_emotion_above_threshold(self):
        emotion = EmotionState(dimensions={"慌乱": 0.5})
        assert derive_engagement_mode(emotion) == EngagementMode.FLUSTERED

    def test_flustered_overrides_heavy(self):
        emotion = EmotionState(dimensions={"慌乱": 0.6, "悲伤": 0.7})
        assert derive_engagement_mode(emotion) == EngagementMode.FLUSTERED

    def test_flustered_overrides_provoked(self):
        emotion = EmotionState(dimensions={"慌乱": 0.5, "不爽": 0.5})
        assert derive_engagement_mode(emotion) == EngagementMode.FLUSTERED

    def test_flustered_below_threshold_falls_through(self):
        emotion = EmotionState(dimensions={"慌乱": 0.2, "悲伤": 0.6})
        assert derive_engagement_mode(emotion) == EngagementMode.WITHDRAWN


class TestPromptRendering:
    def _persona(self):
        return PersonaConfig(name="T", identity=Identity(full_name="T"))

    def test_full_mode_renders_nothing(self):
        builder = PromptBuilder(self._persona())
        reminder = builder.build_turn_focus_reminder() or ""
        assert "不太想多聊" not in reminder
        assert "心里压着事" not in reminder
        assert "心慌了一下" not in reminder

    def test_withdrawn_renders_section(self):
        builder = PromptBuilder(self._persona())
        emotion = EmotionState(dimensions={"悲伤": 0.7})
        prompt = builder.build_turn_focus_reminder(emotion_state=emotion) or ""
        assert "心里压着事" in prompt
        assert "沉默是一等选项" in prompt
        assert "按你此刻的样子写" in prompt

    def test_curt_renders_section(self):
        builder = PromptBuilder(self._persona())
        emotion = EmotionState(dimensions={"不爽": 0.6})
        prompt = builder.build_turn_focus_reminder(emotion_state=emotion) or ""
        assert "不太想多聊" in prompt
        assert "短就够" in prompt
        assert "心里压着事" not in prompt

    def test_withdrawn_excludes_curt_text(self):
        builder = PromptBuilder(self._persona())
        emotion = EmotionState(dimensions={"悲伤": 0.7})
        prompt = builder.build_turn_focus_reminder(emotion_state=emotion) or ""
        assert "不太想多聊" not in prompt

    def test_flustered_renders_section(self):
        builder = PromptBuilder(self._persona())
        emotion = EmotionState(dimensions={"慌乱": 0.6, "心虚": 0.4})
        prompt = builder.build_turn_focus_reminder(emotion_state=emotion) or ""
        assert "心慌了一下" in prompt
        assert "不完整" in prompt
        assert "字重复" in prompt
        assert "过度解释" in prompt
        assert "通顺" in prompt
        assert "焦点留在" in prompt
        assert "时间/天气/具体数字" in prompt
