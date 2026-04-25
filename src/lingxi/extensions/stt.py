"""STT extension interface for future speech-to-text integration."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class STTConfig:
    """Configuration for speech recognition."""

    language: str = "zh-CN"
    model: str = "default"
    enable_punctuation: bool = True


class STTExtension(ABC):
    """Abstract interface for STT extensions."""

    @abstractmethod
    async def transcribe(self, audio_data: bytes, config: STTConfig) -> str:
        """Transcribe audio data to text."""

    @abstractmethod
    async def start_stream(self, config: STTConfig):
        """Start a streaming transcription session."""

    @abstractmethod
    async def feed_audio(self, audio_chunk: bytes) -> str | None:
        """Feed audio data to the stream. Returns transcribed text if available."""

    @abstractmethod
    async def stop_stream(self) -> str:
        """Stop streaming and return final transcription."""
