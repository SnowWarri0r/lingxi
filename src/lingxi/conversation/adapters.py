"""Channel adapters: extract channel-appropriate view of a TurnOutput.

Each channel takes what it can render/use and drops the rest.
This lets Aria produce a rich multi-modal output while each channel
only shows its slice.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from lingxi.conversation.output_schema import TurnOutput


class OutputAdapter(ABC):
    """Base class for channel-specific output adapters."""

    @abstractmethod
    def adapt(self, output: TurnOutput) -> object:
        """Return channel-specific representation of a TurnOutput."""


class TextAdapter(OutputAdapter):
    """Plain-text adapter: returns just the speech.

    Used by Feishu, CLI, Web WebSocket — any channel that only shows text.
    """

    def adapt(self, output: TurnOutput) -> str:
        return output.speech


@dataclass
class VoicePayload:
    """What a TTS channel needs."""

    text: str
    mood: str = ""
    emotion_deltas: dict[str, float] = field(default_factory=dict)
    # Voice tone hint for TTS providers that support it
    tone_hint: str = ""


class VoiceAdapter(OutputAdapter):
    """Adapter for TTS channels.

    Strips speech for synthesis, passes mood so provider can adjust tone.
    """

    def adapt(self, output: TurnOutput) -> VoicePayload:
        tone_hint = _mood_to_tone(output.mood_label, output.emotion_deltas)
        return VoicePayload(
            text=output.speech,
            mood=output.mood_label,
            emotion_deltas=dict(output.emotion_deltas),
            tone_hint=tone_hint,
        )


@dataclass
class AvatarPayload:
    """What a Live2D / avatar channel needs."""

    expression: str = ""
    action: str = ""
    mood: str = ""
    emotion_deltas: dict[str, float] = field(default_factory=dict)


class AvatarAdapter(OutputAdapter):
    """Adapter for Live2D / visual avatar rendering.

    Takes expression and action only — doesn't care about text content.
    """

    def adapt(self, output: TurnOutput) -> AvatarPayload:
        return AvatarPayload(
            expression=output.expression,
            action=output.action,
            mood=output.mood_label,
            emotion_deltas=dict(output.emotion_deltas),
        )


@dataclass
class FullPayload:
    """Everything — for debugging, logs, rich UI."""

    speech: str
    expression: str
    action: str
    mood: str
    emotion_deltas: dict[str, float]
    memory_writes: list[str]
    plan_updates: list[str]
    inner_thought: str


class FullAdapter(OutputAdapter):
    """Dump everything."""

    def adapt(self, output: TurnOutput) -> FullPayload:
        return FullPayload(
            speech=output.speech,
            expression=output.expression,
            action=output.action,
            mood=output.mood_label,
            emotion_deltas=dict(output.emotion_deltas),
            memory_writes=list(output.memory_writes),
            plan_updates=list(output.plan_updates),
            inner_thought=output.inner_thought,
        )


def _mood_to_tone(mood: str, deltas: dict[str, float]) -> str:
    """Very rough mapping from mood label + deltas → TTS tone hint."""
    if not mood and not deltas:
        return "neutral"
    dominant = ""
    if deltas:
        dominant = max(deltas.items(), key=lambda kv: kv[1])[0]

    # Map common Chinese emotion words to tone categories
    positive = {"喜悦", "兴奋", "温暖", "期待", "满足", "好奇"}
    negative = {"悲伤", "焦虑", "失望", "孤独", "疲惫", "愤怒"}
    calm = {"平静", "沉思"}

    label = dominant or mood
    if label in positive:
        return "warm"
    if label in negative:
        return "soft-sad"
    if label in calm:
        return "soft"
    return "neutral"
