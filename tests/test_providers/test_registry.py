"""Tests for the provider registry."""

import pytest

from lingxi.providers.registry import ProviderRegistry
from lingxi.providers.claude import ClaudeProvider
from lingxi.providers.openai_provider import OpenAIProvider


class TestProviderRegistry:
    def test_register_and_create(self):
        ProviderRegistry.register_defaults()
        # Just test registration works - actual creation needs API keys
        assert "claude" in ProviderRegistry._llm_providers
        assert "openai" in ProviderRegistry._llm_providers

    def test_unknown_provider(self):
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            ProviderRegistry.create_llm("nonexistent")
