"""Avatar extension interface for future Live2D / visual avatar integration."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class AvatarState:
    """Current state of the avatar."""

    expression: str = "neutral"
    motion: str = "idle"
    lip_sync: bool = False
    parameters: dict = field(default_factory=dict)


@dataclass
class EmotionMapping:
    """Maps emotional states to avatar expressions and motions."""

    emotion: str
    expression: str
    motion: str = "idle"


class AvatarExtension(ABC):
    """Abstract interface for avatar/Live2D extensions."""

    @abstractmethod
    async def set_expression(self, expression: str) -> None:
        """Set the avatar's facial expression."""

    @abstractmethod
    async def set_motion(self, motion: str) -> None:
        """Trigger an avatar motion/animation."""

    @abstractmethod
    async def start_lip_sync(self, audio_data: bytes | None = None) -> None:
        """Start lip sync animation, optionally synced to audio."""

    @abstractmethod
    async def stop_lip_sync(self) -> None:
        """Stop lip sync animation."""

    @abstractmethod
    async def on_emotion_change(self, emotion: str) -> None:
        """React to an emotion change from the conversation engine."""

    @abstractmethod
    async def get_state(self) -> AvatarState:
        """Get the current avatar state."""

    def get_default_emotion_mappings(self) -> list[EmotionMapping]:
        """Default emotion-to-expression mappings. Override for custom models."""
        return [
            EmotionMapping("neutral", "normal"),
            EmotionMapping("happy", "smile"),
            EmotionMapping("sad", "sad"),
            EmotionMapping("angry", "angry"),
            EmotionMapping("surprised", "surprised"),
            EmotionMapping("thinking", "thinking"),
            EmotionMapping("curious", "interested"),
            EmotionMapping("warm", "gentle_smile"),
        ]
