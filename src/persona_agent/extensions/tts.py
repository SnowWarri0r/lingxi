"""TTS extension interface for future text-to-speech integration."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class VoiceConfig:
    """Configuration for voice synthesis."""

    voice_id: str = "default"
    speed: float = 1.0
    pitch: float = 1.0
    emotion: str = "neutral"
    language: str = "zh-CN"


class TTSExtension(ABC):
    """Abstract interface for TTS extensions."""

    @abstractmethod
    async def on_response(self, text: str, voice_config: VoiceConfig) -> bytes:
        """Called after a response is generated to synthesize speech."""

    @abstractmethod
    async def on_response_stream(self, text_chunk: str, voice_config: VoiceConfig):
        """Called for each chunk in streaming mode."""

    def update_voice_for_emotion(self, config: VoiceConfig, emotion: str) -> VoiceConfig:
        """Adjust voice parameters based on emotional state."""
        config.emotion = emotion
        # Subclasses can override for provider-specific adjustments
        return config
