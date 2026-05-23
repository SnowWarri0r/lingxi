"""Tests for derive_engagement_mode + prompt rendering.

Engagement mode is the agency lever — without it, Aria's prompt rules
about "可以敷衍" are soft hints the LLM ignores. With it, withdrawn ==
"engaging fully is the wrong answer", a hard structural switch.
"""

from lingxi.inner_life.models import (
    EngagementMode,
    InnerState,
    derive_engagement_mode,
)
from lingxi.persona.models import EmotionState, Identity, PersonaConfig
from lingxi.persona.prompt_builder import PromptBuilder


class TestDerivation:
    def test_no_state_returns_full(self):
        assert derive_engagement_mode(None, None) == EngagementMode.FULL

    def test_default_inner_state_full(self):
        # Default energy 0.7, no emotion → full
        assert derive_engagement_mode(InnerState(), None) == EngagementMode.FULL

    def test_heavy_emotion_withdraws(self):
        # 悲伤 0.6 → withdrawn (HEAVY family ≥ 0.5)
        emotion = EmotionState(dimensions={"悲伤": 0.6})
        assert (
            derive_engagement_mode(InnerState(), emotion) == EngagementMode.WITHDRAWN
        )

    def test_孤独_above_threshold_withdraws(self):
        emotion = EmotionState(dimensions={"孤独": 0.7})
        assert (
            derive_engagement_mode(InnerState(), emotion) == EngagementMode.WITHDRAWN
        )

    def test_heavy_below_threshold_no_withdraw(self):
        # HEAVY but below 0.5 — stays full unless other triggers
        emotion = EmotionState(dimensions={"悲伤": 0.3})
        # Default energy 0.7 + nothing else → full
        assert derive_engagement_mode(InnerState(), emotion) == EngagementMode.FULL

    def test_provoked_emotion_curt(self):
        # 不爽 0.5 → curt (PROVOKED ≥ 0.4, no HEAVY)
        emotion = EmotionState(dimensions={"不爽": 0.5})
        assert derive_engagement_mode(InnerState(), emotion) == EngagementMode.CURT

    def test_low_energy_curt(self):
        # energy 0.2 < 0.3 → curt
        state = InnerState(energy=0.2)
        assert derive_engagement_mode(state, None) == EngagementMode.CURT

    def test_heavy_overrides_provoked(self):
        # 悲伤 0.6 + 不爽 0.5 → withdrawn (heavy wins)
        emotion = EmotionState(dimensions={"悲伤": 0.6, "不爽": 0.5})
        assert (
            derive_engagement_mode(InnerState(), emotion) == EngagementMode.WITHDRAWN
        )

    def test_heavy_overrides_low_energy(self):
        # Low energy + heavy emotion → withdrawn (heavy wins over curt)
        emotion = EmotionState(dimensions={"悲伤": 0.6})
        state = InnerState(energy=0.1)
        assert (
            derive_engagement_mode(state, emotion) == EngagementMode.WITHDRAWN
        )

    def test_normal_energy_normal_emotion_full(self):
        emotion = EmotionState(dimensions={"平静": 0.5, "好奇": 0.3})
        state = InnerState(energy=0.7)
        assert derive_engagement_mode(state, emotion) == EngagementMode.FULL

    def test_flustered_emotion_above_threshold(self):
        emotion = EmotionState(dimensions={"慌乱": 0.5})
        assert derive_engagement_mode(InnerState(), emotion) == EngagementMode.FLUSTERED

    def test_flustered_overrides_heavy(self):
        # Both flustered and heavy active — flustered dominates because
        # it's the most acute reaction shaping output style.
        emotion = EmotionState(dimensions={"慌乱": 0.6, "悲伤": 0.7})
        assert derive_engagement_mode(InnerState(), emotion) == EngagementMode.FLUSTERED

    def test_flustered_overrides_provoked(self):
        emotion = EmotionState(dimensions={"慌乱": 0.5, "不爽": 0.5})
        assert derive_engagement_mode(InnerState(), emotion) == EngagementMode.FLUSTERED

    def test_flustered_below_threshold_falls_through(self):
        emotion = EmotionState(dimensions={"慌乱": 0.2, "悲伤": 0.6})
        # 慌乱 too low → heavy wins
        assert derive_engagement_mode(InnerState(), emotion) == EngagementMode.WITHDRAWN


class TestPromptRendering:
    def _persona(self):
        return PersonaConfig(name="T", identity=Identity(full_name="T"))

    def test_full_mode_renders_nothing(self):
        builder = PromptBuilder(self._persona())
        # Phase 2: engagement section lives in focus reminder
        reminder = builder.build_turn_focus_reminder(inner_state=InnerState()) or ""
        # Default state → full → no engagement section (no mode-specific headers)
        assert "不太想多聊" not in reminder
        assert "心里压着事" not in reminder
        assert "心慌了一下" not in reminder

    def test_withdrawn_renders_section(self):
        builder = PromptBuilder(self._persona())
        emotion = EmotionState(dimensions={"悲伤": 0.7})
        prompt = builder.build_turn_focus_reminder(
            inner_state=InnerState(),
            emotion_state=emotion,
        ) or ""
        assert "心里压着事" in prompt
        assert "沉默是一等选项" in prompt
        assert "完整周到" in prompt

    def test_curt_renders_section(self):
        builder = PromptBuilder(self._persona())
        emotion = EmotionState(dimensions={"不爽": 0.6})
        prompt = builder.build_turn_focus_reminder(
            inner_state=InnerState(),
            emotion_state=emotion,
        ) or ""
        assert "不太想多聊" in prompt
        assert "短就够" in prompt
        assert "心里压着事" not in prompt  # don't bleed into withdrawn copy

    def test_withdrawn_excludes_curt_text(self):
        # Make sure modes don't overlap in rendering
        builder = PromptBuilder(self._persona())
        emotion = EmotionState(dimensions={"悲伤": 0.7})
        prompt = builder.build_turn_focus_reminder(
            inner_state=InnerState(),
            emotion_state=emotion,
        ) or ""
        assert "不太想多聊" not in prompt  # CURT header

    def test_flustered_renders_section(self):
        builder = PromptBuilder(self._persona())
        emotion = EmotionState(dimensions={"慌乱": 0.6, "心虚": 0.4})
        prompt = builder.build_turn_focus_reminder(
            inner_state=InnerState(),
            emotion_state=emotion,
        ) or ""
        assert "心慌了一下" in prompt
        # Critical anti-composure phrasings
        assert "不完整" in prompt
        assert "字重复" in prompt
        assert "过度解释" in prompt
        # Anti-garbled-grammar guard (carbon vs garbled distinction)
        assert "乱码不通" in prompt
        assert "通顺" in prompt
        # Anti-topic-switch guard
        assert "无关问题转移焦点" in prompt or "无关问题" in prompt
        # Anti-fabrication guard (look at system-reminder facts, don't invent)
        assert "时间/天气/具体数字" in prompt or "不知道就别说" in prompt
