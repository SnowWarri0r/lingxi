"""Abstract interfaces for external service providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator


@dataclass
class CompletionResult:
    """Result from an LLM completion call."""

    content: str
    model: str = ""
    usage: dict = field(default_factory=dict)
    finish_reason: str = ""


@dataclass
class StreamChunk:
    """A single chunk from a streaming completion."""

    content: str
    is_final: bool = False


@dataclass
class AudioData:
    """Container for audio data."""

    data: bytes
    sample_rate: int = 24000
    format: str = "pcm"


@dataclass
class TranscriptionResult:
    """Result from speech-to-text."""

    text: str
    confidence: float = 1.0
    language: str = ""


class LLMProvider(ABC):
    """Abstract interface for LLM providers."""

    @abstractmethod
    async def complete(
        self,
        messages: list[dict],
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float = 1.0,
        prefill: str = "",
        **kwargs,
    ) -> CompletionResult:
        """Generate a completion from messages."""

    async def complete_stream(
        self,
        messages: list[dict],
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float = 1.0,
        prefill: str = "",
        **kwargs,
    ) -> AsyncIterator[StreamChunk]:
        """Stream a completion. Default implementation wraps complete()."""
        result = await self.complete(
            messages, system, max_tokens, temperature, top_p, prefill, **kwargs,
        )
        yield StreamChunk(content=result.content, is_final=True)

    async def embed(self, text: str) -> list[float]:
        """Generate an embedding vector for the text. Optional."""
        raise NotImplementedError("This provider does not support embeddings")


class TTSProvider(ABC):
    """Abstract interface for text-to-speech providers."""

    @abstractmethod
    async def synthesize(
        self,
        text: str,
        voice: str = "default",
        speed: float = 1.0,
        **kwargs,
    ) -> AudioData:
        """Convert text to speech audio."""

    async def synthesize_stream(
        self,
        text: str,
        voice: str = "default",
        speed: float = 1.0,
        **kwargs,
    ) -> AsyncIterator[AudioData]:
        """Stream audio synthesis. Default wraps synthesize()."""
        result = await self.synthesize(text, voice, speed, **kwargs)
        yield result


class STTProvider(ABC):
    """Abstract interface for speech-to-text providers."""

    @abstractmethod
    async def transcribe(self, audio: AudioData, **kwargs) -> TranscriptionResult:
        """Transcribe audio to text."""
