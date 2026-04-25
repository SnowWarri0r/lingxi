"""OpenAI LLM provider implementation."""

from __future__ import annotations

from typing import AsyncIterator

import openai

from lingxi.providers.base import CompletionResult, LLMProvider, StreamChunk


class OpenAIProvider(LLMProvider):
    """LLM provider using OpenAI's API.

    Supports both direct API key and OAuth token authentication.
    """

    def __init__(
        self,
        api_key: str = "",
        model: str = "gpt-4o",
        base_url: str | None = None,
    ):
        self.model = model
        self._api_key = api_key
        self._base_url = base_url
        self._client: openai.AsyncOpenAI | None = None

    def _get_client(self) -> openai.AsyncOpenAI:
        if self._client is None:
            self._client = openai.AsyncOpenAI(api_key=self._api_key, base_url=self._base_url)
        return self._client

    def update_credentials(self, api_key: str) -> None:
        """Update credentials (e.g., after OAuth login or token refresh)."""
        self._api_key = api_key
        self._client = None

    async def complete(
        self,
        messages: list[dict],
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs,
    ) -> CompletionResult:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.extend(messages)

        response = await self._get_client().chat.completions.create(
            model=self.model,
            messages=msgs,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        choice = response.choices[0]
        return CompletionResult(
            content=choice.message.content or "",
            model=response.model or self.model,
            usage={
                "input_tokens": response.usage.prompt_tokens if response.usage else 0,
                "output_tokens": response.usage.completion_tokens if response.usage else 0,
            },
            finish_reason=choice.finish_reason or "",
        )

    async def complete_stream(
        self,
        messages: list[dict],
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs,
    ) -> AsyncIterator[StreamChunk]:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.extend(messages)

        stream = await self._get_client().chat.completions.create(
            model=self.model,
            messages=msgs,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield StreamChunk(content=chunk.choices[0].delta.content)

        yield StreamChunk(content="", is_final=True)

    async def embed(self, text: str) -> list[float]:
        """Generate embeddings using OpenAI's embedding API."""
        response = await self._get_client().embeddings.create(
            model="text-embedding-3-small",
            input=text,
        )
        return response.data[0].embedding
