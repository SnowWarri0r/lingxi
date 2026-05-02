"""Pydantic models for persona definition."""

from __future__ import annotations

import math
from datetime import datetime

from pydantic import BaseModel, Field


class Trait(BaseModel):
    trait: str
    intensity: float = Field(ge=0.0, le=1.0, description="Trait intensity from 0 to 1")


class Identity(BaseModel):
    full_name: str
    age: int | None = None
    occupation: str | None = None
    background: str = ""


class PersonalityProfile(BaseModel):
    traits: list[Trait] = Field(default_factory=list)
    values: list[str] = Field(default_factory=list)
    fears: list[str] = Field(default_factory=list)


class SpeakingStyle(BaseModel):
    tone: str = "neutral"
    vocabulary_level: str = "normal"
    verbal_habits: list[str] = Field(default_factory=list)
    example_phrases: list[str] = Field(default_factory=list)


class EmotionalTrigger(BaseModel):
    mood: str
    trigger: str


class EmotionalBaseline(BaseModel):
    default_mood: str = "neutral"
    mood_volatility: float = Field(default=0.5, ge=0.0, le=1.0)
    emotional_range: list[EmotionalTrigger] = Field(default_factory=list)
    # Resting state: dimension_name -> baseline intensity (0-1)
    baseline_dimensions: dict[str, float] = Field(default_factory=dict)


class EmotionState(BaseModel):
    """Multi-dimensional emotional state with decay toward baseline.

    dimensions: {name: intensity}, each intensity in [0, 1]
    narrative_label: free-form mood string (from <mood_update>)
    """

    dimensions: dict[str, float] = Field(default_factory=dict)
    narrative_label: str = ""
    last_decay_at: datetime = Field(default_factory=datetime.now)

    @classmethod
    def from_baseline(cls, baseline: EmotionalBaseline) -> EmotionState:
        dims = dict(baseline.baseline_dimensions) if baseline.baseline_dimensions else {}
        # Auto-populate a few defaults if none provided
        if not dims:
            dims = {"平静": 0.5, "好奇": 0.3}
        return cls(
            dimensions=dict(dims),
            narrative_label=baseline.default_mood,
        )

    def apply_deltas(self, deltas: dict[str, float], volatility: float = 0.5) -> None:
        """Apply delta-based updates scaled by persona volatility.

        deltas: {dimension_name: new_intensity_or_delta}
        If the value looks like an absolute intensity (0-1), set it directly
        (scaled by volatility toward the current value).
        """
        for name, target in deltas.items():
            try:
                target_val = float(target)
            except (TypeError, ValueError):
                continue
            target_val = max(0.0, min(1.0, target_val))
            current = self.dimensions.get(name, 0.0)
            # Blend: more volatile personas jump faster to target
            new_val = current + (target_val - current) * (0.3 + 0.7 * volatility)
            self.dimensions[name] = max(0.0, min(1.0, new_val))

    def decay_toward_baseline(
        self,
        baseline: dict[str, float],
        decay_rate: float = 0.0001,
    ) -> None:
        """Exponential decay each dimension toward its baseline.

        decay_rate: per-second decay rate. Default 0.0001/s ≈ half-life ~115min.
        """
        now = datetime.now()
        seconds = (now - self.last_decay_at).total_seconds()
        if seconds <= 0:
            return

        factor = math.exp(-decay_rate * seconds)
        new_dims: dict[str, float] = {}

        # Current dimensions decay toward their baseline (or 0 if no baseline)
        for name, value in self.dimensions.items():
            base = baseline.get(name, 0.0)
            decayed = base + (value - base) * factor
            if decayed > 0.02:  # Prune very small values
                new_dims[name] = round(decayed, 3)

        # Ensure baseline dims are present (they rise back toward baseline over time)
        for name, base in baseline.items():
            if name not in new_dims and base > 0.02:
                current = self.dimensions.get(name, 0.0)
                decayed = base + (current - base) * factor
                if decayed > 0.02:
                    new_dims[name] = round(decayed, 3)

        self.dimensions = new_dims
        self.last_decay_at = now

    def dominant(self) -> tuple[str, float] | None:
        if not self.dimensions:
            return None
        name, val = max(self.dimensions.items(), key=lambda kv: kv[1])
        return name, val

    def top_k(self, k: int = 3, min_intensity: float = 0.15) -> list[tuple[str, float]]:
        items = [(n, v) for n, v in self.dimensions.items() if v >= min_intensity]
        items.sort(key=lambda kv: kv[1], reverse=True)
        return items[:k]

    def overall_valence(self) -> str:
        """Rough 'positive/negative/neutral' classification based on dimensions."""
        positive = {"喜悦", "温暖", "兴奋", "好奇", "平静", "满足", "期待", "感动"}
        negative = {"悲伤", "愤怒", "焦虑", "失望", "孤独", "不安", "疲惫"}

        pos_sum = sum(v for n, v in self.dimensions.items() if n in positive)
        neg_sum = sum(v for n, v in self.dimensions.items() if n in negative)

        if pos_sum > neg_sum + 0.3:
            return "积极"
        if neg_sum > pos_sum + 0.3:
            return "消极"
        return "平和"

    def to_prompt_text(self) -> str:
        """Render the emotion state for the system prompt."""
        top = self.top_k(k=3, min_intensity=0.2)
        if not top:
            return "情绪平稳"

        main_name, main_val = top[0]
        parts = [f"你现在主要感到 **{main_name}**（强度 {main_val:.1f}）"]

        if len(top) > 1:
            extras = "、".join(f"{n}({v:.1f})" for n, v in top[1:])
            parts.append(f"，同时伴有 {extras}")

        parts.append(f"。整体情绪基调：{self.overall_valence()}")

        if self.narrative_label:
            parts.append(f"（关键词：{self.narrative_label}）")

        return "".join(parts)


class StyleConfig(BaseModel):
    """Output style controls for the single-call combo."""

    speech_max_chars: int = Field(default=40, ge=1, le=500)
    prefill_openers: list[str] = Field(
        default_factory=lambda: ["嗯", "欸", "哦", ""],
        description="Random-pick prefill for assistant message; empty string = no prefill.",
    )
    blacklist_phrases: list[str] = Field(
        default_factory=list,
        description="Extra phrases to ban, on top of the code-default list.",
    )


class SamplingConfig(BaseModel):
    """LLM sampling parameters."""

    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)


class CompressionConfig(BaseModel):
    """Two-call (think → compress-to-speech) pipeline.

    When enabled, every turn runs:
      1. Think (main model): persona + memory + biography → inner_thought + meta
      2. Compress (smaller model): inner_thought + few-shot + voice → speech

    Trade-off: ~2× total latency (TTFB ~2s vs ~1s with stage UI), but the
    LLM in Call 2 sees a "transcribe a thought" task instead of "answer a
    user", which is outside the helpful-assistant distribution and produces
    much less AI-tone.
    """

    enabled: bool = False
    compress_model: str = "claude-haiku-4-5-20251001"
    compress_max_tokens: int = 200
    compress_temperature: float = 0.9


class GoalDefinition(BaseModel):
    description: str
    priority: float = Field(default=0.5, ge=0.0, le=1.0)


class IntimacyLevel(BaseModel):
    level: int
    name: str
    description: str


class Relationship(BaseModel):
    initial_stance: str = "friendly"
    trust_building: str = ""
    intimacy_levels: list[IntimacyLevel] = Field(default_factory=list)


class LifeEvent(BaseModel):
    """A single past event/memory from the persona's life."""

    age: int = Field(ge=0, le=150)
    content: str
    tags: list[str] = Field(default_factory=list)


class RecurringPerson(BaseModel):
    """A person who recurs in the persona's life (family, close friend, mentor)."""

    name: str
    relation: str  # one-liner: "教我认星座的人，去年因癌症离世"


class Biography(BaseModel):
    """Past life material — referenced by retrieval, not always injected."""

    life_events: list[LifeEvent] = Field(default_factory=list)
    recurring_people: list[RecurringPerson] = Field(default_factory=list)
    motifs: list[str] = Field(default_factory=list)  # recurring images / objects


class PersonaConfig(BaseModel):
    """Root configuration for a virtual persona."""

    name: str
    version: str = "1.0"
    identity: Identity
    personality: PersonalityProfile = Field(default_factory=PersonalityProfile)
    speaking_style: SpeakingStyle = Field(default_factory=SpeakingStyle)
    emotional_baseline: EmotionalBaseline = Field(default_factory=EmotionalBaseline)
    goals: list[GoalDefinition] = Field(default_factory=list)
    relationship: Relationship = Field(default_factory=Relationship)
    style: StyleConfig = Field(default_factory=StyleConfig)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    biography: Biography = Field(default_factory=Biography)
    compression: CompressionConfig = Field(default_factory=CompressionConfig)
