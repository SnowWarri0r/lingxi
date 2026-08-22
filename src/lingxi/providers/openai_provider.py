"""OpenAI LLM provider implementation."""

from __future__ import annotations

import time
from typing import AsyncIterator

import openai

from lingxi.providers.base import CompletionResult, LLMProvider, StreamChunk


def _log(system, messages, text, model, usage, started, purpose):
    """Record the call, when LINGXI_DEBUG_LLM is on.

    The responder was the one provider that never logged. That gap only
    showed itself weeks later: a failure measured live on 2026-08-19 could
    not be reproduced afterwards, and with no record of what had actually
    been sent or returned there was nothing to diff against — the three
    candidate explanations stayed candidates. It is the biggest prompt on
    every turn and the one whose output the user actually reads.
    """
    try:
        from lingxi.debug.request_log import log_request
        log_request(
            system=system or "", messages=messages, response_text=text,
            model=model, usage=usage,
            duration_ms=int((time.monotonic() - started) * 1000),
            purpose=purpose,
        )
    except Exception:
        pass  # debug logging must never break a turn


def _cached_tokens(u) -> int:
    """Prefix-cache hits, however this vendor spells them.

    DeepSeek: prompt_cache_hit_tokens. Doubao/ARK: prompt_tokens_details
    .cached_tokens. Both were being dropped, which left the responder's cache
    behaviour — the single biggest prompt on every turn — unobservable.
    """
    hit = getattr(u, "prompt_cache_hit_tokens", None)
    if hit is not None:
        return hit or 0
    details = getattr(u, "prompt_tokens_details", None)
    return (getattr(details, "cached_tokens", 0) or 0) if details else 0


def _usage(u) -> dict:
    if u is None:
        return {"input_tokens": 0, "output_tokens": 0}
    return {
        "input_tokens": u.prompt_tokens or 0,
        "output_tokens": u.completion_tokens or 0,
        "cache_read_tokens": _cached_tokens(u),
    }


class OpenAIProvider(LLMProvider):
    """LLM provider using OpenAI's API.

    Supports both direct API key and OAuth token authentication.
    """

    def __init__(
        self,
        api_key: str = "",
        model: str = "gpt-4o",
        base_url: str | None = None,
        extra_body: dict | None = None,
        report_usage: bool = False,
    ):
        self.model = model
        self._api_key = api_key
        self._base_url = base_url
        # Ask the vendor for token/cache counts on streamed calls and log them.
        # Off by default so an untested OpenAI-compatible endpoint can't reject
        # the request over a parameter it doesn't know.
        self._report_usage = report_usage
        # Vendor-specific params merged into every create() call (e.g. doubao's
        # {"thinking": {"type": "disabled"}} to skip the multi-second reasoning
        # phase so chat replies start streaming immediately).
        self._extra_body = extra_body or {}
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
        top_p: float | None = None,
        prefill: str = "",
        **kwargs,
    ) -> CompletionResult:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.extend(messages)

        # top_p forwarded to OpenAI chat.completions.create as a native parameter
        # prefill not natively supported by OpenAI chat API
        create_kwargs: dict = dict(
            model=self.model,
            messages=msgs,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if top_p is not None:
            create_kwargs["top_p"] = top_p
        if self._extra_body:
            create_kwargs["extra_body"] = self._extra_body
        started = time.monotonic()
        response = await self._get_client().chat.completions.create(**create_kwargs)

        choice = response.choices[0]
        content = choice.message.content or ""
        usage = _usage(response.usage)
        _log(system, messages, content, response.model or self.model,
             usage, started, kwargs.get("_debug_purpose", "responder"))
        return CompletionResult(
            content=content,
            model=response.model or self.model,
            usage=usage,
            finish_reason=choice.finish_reason or "",
        )

    async def complete_stream(
        self,
        messages: list[dict],
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float | None = None,
        prefill: str = "",
        **kwargs,
    ) -> AsyncIterator[StreamChunk]:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.extend(messages)

        # top_p forwarded to OpenAI chat.completions.create as a native parameter
        # prefill not natively supported by OpenAI chat API
        create_kwargs: dict = dict(
            model=self.model,
            messages=msgs,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )
        if top_p is not None:
            create_kwargs["top_p"] = top_p
        if self._extra_body:
            create_kwargs["extra_body"] = self._extra_body
        # Usage only arrives on a stream when asked for. Verified against both
        # api.deepseek.com and ark.cn-beijing.volces.com before enabling.
        if self._report_usage:
            create_kwargs["stream_options"] = {"include_usage": True}
        started = time.monotonic()
        stream = await self._get_client().chat.completions.create(**create_kwargs)

        usage = None
        collected = ""
        async for chunk in stream:
            if getattr(chunk, "usage", None):
                usage = chunk.usage
            if chunk.choices and chunk.choices[0].delta.content:
                piece = chunk.choices[0].delta.content
                collected += piece
                yield StreamChunk(content=piece)

        _log(system, messages, collected, self.model, _usage(usage), started,
             kwargs.get("_debug_purpose", "responder_stream"))

        if usage is not None:
            total = usage.prompt_tokens or 0
            hit = _cached_tokens(usage)
            pct = (hit / total * 100) if total else 0.0
            print(f"[cache] {self.model} prompt={total} cached={hit} "
                  f"({pct:.0f}%)", flush=True)

        yield StreamChunk(content="", is_final=True)

    async def embed(self, text: str) -> list[float]:
        """Generate embeddings using OpenAI's embedding API."""
        response = await self._get_client().embeddings.create(
            model="text-embedding-3-small",
            input=text,
        )
        return response.data[0].embedding
