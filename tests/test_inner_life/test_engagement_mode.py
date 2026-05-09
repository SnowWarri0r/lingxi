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
        # Default state → full → no engagement section
        assert "CURT 模式" not in reminder
        assert "WITHDRAWN 模式" not in reminder

    def test_withdrawn_renders_section(self):
        builder = PromptBuilder(self._persona())
        emotion = EmotionState(dimensions={"悲伤": 0.7})
        prompt = builder.build_turn_focus_reminder(
            inner_state=InnerState(),
            emotion_state=emotion,
        ) or ""
        assert "WITHDRAWN 模式" in prompt
        # Critical phrasings from the section
        assert "沉默是一等选项" in prompt
        assert "完整 / 周到" in prompt or "完整" in prompt

    def test_curt_renders_section(self):
        builder = PromptBuilder(self._persona())
        emotion = EmotionState(dimensions={"不爽": 0.6})
        prompt = builder.build_turn_focus_reminder(
            inner_state=InnerState(),
            emotion_state=emotion,
        ) or ""
        assert "CURT 模式" in prompt
        assert "不接得圆" in prompt
        assert "WITHDRAWN" not in prompt

    def test_withdrawn_excludes_curt_text(self):
        # Make sure modes don't overlap in rendering
        builder = PromptBuilder(self._persona())
        emotion = EmotionState(dimensions={"悲伤": 0.7})
        prompt = builder.build_turn_focus_reminder(
            inner_state=InnerState(),
            emotion_state=emotion,
        ) or ""
        assert "CURT 模式" not in prompt

    def test_flustered_renders_section(self):
        builder = PromptBuilder(self._persona())
        emotion = EmotionState(dimensions={"慌乱": 0.6, "心虚": 0.4})
        prompt = builder.build_turn_focus_reminder(
            inner_state=InnerState(),
            emotion_state=emotion,
        ) or ""
        assert "FLUSTERED 模式" in prompt
        # Critical anti-composure phrasings
        assert "不完整" in prompt
        assert "重复" in prompt
        assert "过度解释" in prompt
        # Anti-garbled-grammar guard
        assert "语法不通" in prompt or "乱码" in prompt
        assert "不完整 ≠ 不通顺" in prompt
        # Anti-topic-switch guard
        assert "换话题" in prompt
        assert "你现在还在工作吗" in prompt or "无关的 follow-up" in prompt
        # Anti-fabrication guard (don't invent times/numbers as filler)
        assert "凭空塞" in prompt or "编个事实" in prompt
        assert "快五点了" in prompt
