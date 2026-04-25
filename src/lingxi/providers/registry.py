"""Provider registry: discover and instantiate providers by config name."""

from __future__ import annotations

from typing import Any

from lingxi.auth.manager import AuthManager
from lingxi.auth.models import AuthConfig, AuthMethod
from lingxi.providers.base import LLMProvider, STTProvider, TTSProvider


# Maps registry name -> (provider name for auth, env var fallback)
_AUTH_PROVIDER_MAP: dict[str, str] = {
    "claude": "anthropic",
    "openai": "openai",
}


class ProviderRegistry:
    """Registry for discovering and instantiating service providers.

    Supports OAuth device flow, cached tokens, and API key authentication.
    """

    _llm_providers: dict[str, type[LLMProvider]] = {}
    _tts_providers: dict[str, type[TTSProvider]] = {}
    _stt_providers: dict[str, type[STTProvider]] = {}

    @classmethod
    def register_llm(cls, name: str, provider_class: type[LLMProvider]) -> None:
        cls._llm_providers[name] = provider_class

    @classmethod
    def register_tts(cls, name: str, provider_class: type[TTSProvider]) -> None:
        cls._tts_providers[name] = provider_class

    @classmethod
    def register_stt(cls, name: str, provider_class: type[STTProvider]) -> None:
        cls._stt_providers[name] = provider_class

    @classmethod
    def create_llm(cls, name: str, **kwargs: Any) -> LLMProvider:
        """Create an LLM provider instance (without resolving auth yet)."""
        if name not in cls._llm_providers:
            raise ValueError(
                f"Unknown LLM provider: {name}. Available: {list(cls._llm_providers.keys())}"
            )
        return cls._llm_providers[name](**kwargs)

    @classmethod
    async def create_llm_with_auth(
        cls,
        name: str,
        auth_manager: AuthManager,
        auth_method: AuthMethod = AuthMethod.OAUTH_DEVICE_FLOW,
        **kwargs: Any,
    ) -> LLMProvider:
        """Create an LLM provider with automatic credential resolution.

        Resolution order:
        1. Cached OAuth token
        2. Environment variable API key
        3. Interactive OAuth device flow login
        """
        if name not in cls._llm_providers:
            raise ValueError(
                f"Unknown LLM provider: {name}. Available: {list(cls._llm_providers.keys())}"
            )

        auth_provider = _AUTH_PROVIDER_MAP.get(name, name)
        auth_config = AuthConfig(provider=auth_provider, method=auth_method)

        # Resolve credentials
        credential = await auth_manager.resolve_credentials(auth_config)

        # Create provider with resolved credentials
        return cls._llm_providers[name](api_key=credential, **kwargs)

    @classmethod
    def create_tts(cls, name: str, **kwargs: Any) -> TTSProvider:
        if name not in cls._tts_providers:
            raise ValueError(
                f"Unknown TTS provider: {name}. Available: {list(cls._tts_providers.keys())}"
            )
        return cls._tts_providers[name](**kwargs)

    @classmethod
    def create_stt(cls, name: str, **kwargs: Any) -> STTProvider:
        if name not in cls._stt_providers:
            raise ValueError(
                f"Unknown STT provider: {name}. Available: {list(cls._stt_providers.keys())}"
            )
        return cls._stt_providers[name](**kwargs)

    @classmethod
    def register_defaults(cls) -> None:
        """Register built-in providers."""
        from lingxi.providers.claude import ClaudeProvider
        from lingxi.providers.openai_provider import OpenAIProvider

        cls.register_llm("claude", ClaudeProvider)
        cls.register_llm("openai", OpenAIProvider)
