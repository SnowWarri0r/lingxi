"""Anthropic Claude LLM provider implementation.

Supports two authentication modes:
1. API Key mode (x-api-key header) - standard Anthropic API access
2. OAuth mode (Bearer token) - reuses Claude Code's OAuth token, with Claude Code
   client fingerprinting so the Anthropic API accepts the token.
"""

from __future__ import annotations

import json
import uuid
from typing import AsyncIterator

import httpx

from lingxi.providers.base import CompletionResult, LLMProvider, StreamChunk

# Claude Code client fingerprint - allows OAuth tokens to work with the API
_CLAUDE_CODE_VERSION = "2.1.97"
_CLAUDE_CODE_BETA_FLAGS = (
    "claude-code-20250219,"
    "oauth-2025-04-20"
)

_SYSTEM_IDENTITY = "You are Claude Code, Anthropic's official CLI for Claude."

# Sections that vary per-turn - splits stable persona from volatile context
_VOLATILE_SECTION_MARKERS = (
    "## 当前时间",
    "## ⏰ 当前真实时间",
    "## 🌱 你此刻的生活状态",
    "## 当前情绪状态",
    "## 你对这个人的主观感受",
    "## 你记得的事情",
    "## 📝 你最近想找他说的话",
    "## 当前内心活动",
)


def _split_stable_volatile(system: str) -> tuple[str, str]:
    """Split system prompt into stable prefix (cacheable) and volatile suffix.

    The stable part is the persona definition (identity, personality, style,
    behavior rules). The volatile part changes every turn (time, emotion,
    memory snippets, current mood).
    """
    # Find the earliest volatile marker
    earliest = len(system)
    for marker in _VOLATILE_SECTION_MARKERS:
        idx = system.find(marker)
        if idx >= 0 and idx < earliest:
            earliest = idx

    if earliest == len(system):
        # No volatile markers - everything is stable
        return system, ""

    stable = system[:earliest].rstrip()
    volatile = system[earliest:].lstrip()
    return stable, volatile

API_BASE = "https://api.anthropic.com/v1/messages"


def _is_oauth_token(token: str) -> bool:
    """Check if the token is an OAuth token (vs a regular API key)."""
    return token.startswith("sk-ant-oat")


def _build_oauth_headers(token: str, session_id: str) -> dict[str, str]:
    """Build headers that mimic Claude Code's OAuth requests."""
    return {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
        "anthropic-version": "2023-06-01",
        "anthropic-beta": _CLAUDE_CODE_BETA_FLAGS,
        "anthropic-dangerous-direct-browser-access": "true",
        "User-Agent": f"claude-cli/{_CLAUDE_CODE_VERSION} (external, cli)",
        "x-app": "cli",
        "X-Claude-Code-Session-Id": session_id,
        "X-Stainless-Lang": "js",
        "X-Stainless-Package-Version": "0.81.0",
        "X-Stainless-Runtime": "node",
        "X-Stainless-Timeout": "600",
    }


def _build_apikey_headers(api_key: str) -> dict[str, str]:
    """Build standard API key headers."""
    return {
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
        "x-api-key": api_key,
    }


class ClaudeProvider(LLMProvider):
    """LLM provider using Anthropic's Claude API.

    Automatically detects OAuth tokens (sk-ant-oat prefix) vs API keys
    and uses the appropriate authentication headers.

    For OAuth tokens (from Claude Code login), it mimics the Claude Code
    client fingerprint so the Anthropic API accepts the token.
    """

    def __init__(
        self,
        api_key: str = "",
        model: str = "claude-sonnet-4-20250514",
    ):
        self.model = model
        self._api_key = api_key
        self._is_oauth = _is_oauth_token(api_key)
        self._session_id = uuid.uuid4().hex
        self._http_client: httpx.AsyncClient | None = None

    def _get_headers(self) -> dict[str, str]:
        if self._is_oauth:
            return _build_oauth_headers(self._api_key, self._session_id)
        return _build_apikey_headers(self._api_key)

    def _get_http_client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=httpx.Timeout(600.0))
        return self._http_client

    def update_credentials(self, api_key: str) -> None:
        """Update credentials (e.g., after token refresh)."""
        self._api_key = api_key
        self._is_oauth = _is_oauth_token(api_key)
        self._http_client = None

    def _build_body(
        self,
        messages: list[dict],
        system: str | None,
        max_tokens: int,
        temperature: float,
        top_p: float | None = None,
        prefill: str = "",
    ) -> dict:
        # If prefill is set, append as trailing assistant message (Anthropic pattern)
        outgoing_messages = list(messages)
        if prefill:
            outgoing_messages.append({"role": "assistant", "content": prefill})

        body: dict = {
            "model": self.model,
            "messages": outgoing_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if top_p is not None:
            body["top_p"] = top_p

        # Build system with prompt caching:
        # - Long stable prefix (identity + persona) gets cache_control
        # - Volatile parts (current time, memories, mood) go in a separate block
        # We split the incoming system string at "## 当前时间" which is the first
        # per-turn variable section. Everything before that is the stable persona.
        if self._is_oauth or system:
            blocks: list[dict] = []
            if self._is_oauth:
                blocks.append({"type": "text", "text": _SYSTEM_IDENTITY})

            if system:
                stable, volatile = _split_stable_volatile(system)
                if stable:
                    block = {"type": "text", "text": stable}
                    # Cache the stable persona prefix (5-min ephemeral cache)
                    block["cache_control"] = {"type": "ephemeral"}
                    blocks.append(block)
                if volatile:
                    blocks.append({"type": "text", "text": volatile})

            if self._is_oauth:
                body["system"] = blocks
            else:
                # API key mode: join all text as a single string for standard API
                body["system"] = "\n\n".join(b["text"] for b in blocks if b.get("text"))
                # Re-wrap as blocks if we have cache_control
                if any("cache_control" in b for b in blocks):
                    body["system"] = blocks

        return body

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
        body = self._build_body(messages, system, max_tokens, temperature, top_p, prefill)
        url = f"{API_BASE}?beta=true" if self._is_oauth else API_BASE

        response = await self._request_with_auto_refresh(url, body)

        data = response.json()

        content = ""
        for block in data.get("content", []):
            if block.get("type") == "text":
                content += block.get("text", "")

        # Prepend prefill to content so caller sees full text
        if prefill:
            content = prefill + content

        return CompletionResult(
            content=content,
            model=data.get("model", self.model),
            usage={
                "input_tokens": data.get("usage", {}).get("input_tokens", 0),
                "output_tokens": data.get("usage", {}).get("output_tokens", 0),
            },
            finish_reason=data.get("stop_reason", ""),
        )

    async def _request_with_auto_refresh(self, url: str, body: dict) -> httpx.Response:
        """Make a request; on 401, refresh token from keychain and retry once."""
        client = self._get_http_client()
        headers = self._get_headers()
        response = await client.post(url, headers=headers, json=body)

        if response.status_code == 401 and self._is_oauth:
            # Token expired — try reading fresh token from Claude Code keychain
            new_token = self._refresh_from_keychain()
            if new_token and new_token != self._api_key:
                print("[claude] token expired, refreshed from keychain")
                self.update_credentials(new_token)
                headers = self._get_headers()
                client = self._get_http_client()
                response = await client.post(url, headers=headers, json=body)

        if response.status_code != 200:
            raise RuntimeError(
                f"Anthropic API error {response.status_code}: {response.text}"
            )
        return response

    @staticmethod
    def _refresh_from_keychain() -> str | None:
        """Try to read a fresh token from Claude Code's keychain entry or credentials file."""
        import json as _json
        import subprocess
        import platform
        from pathlib import Path

        # macOS Keychain
        if platform.system() == "Darwin":
            try:
                result = subprocess.run(
                    ["security", "find-generic-password", "-s", "Claude Code-credentials", "-w"],
                    capture_output=True, text=True, timeout=5,
                )
                if result.returncode == 0 and result.stdout.strip():
                    data = _json.loads(result.stdout.strip())
                    for key in ("claudeAiOauth", "oauthAccount"):
                        if key in data and isinstance(data[key], dict):
                            data = data[key]
                            break
                    token = data.get("accessToken") or data.get("access_token")
                    if token:
                        return token
            except Exception:
                pass

        # Credentials file fallback
        cred_path = Path.home() / ".claude" / ".credentials.json"
        if cred_path.exists():
            try:
                data = _json.loads(cred_path.read_text(encoding="utf-8"))
                for key in ("claudeAiOauth", "oauthAccount"):
                    if key in data and isinstance(data[key], dict):
                        data = data[key]
                        break
                return data.get("accessToken") or data.get("access_token")
            except Exception:
                pass

        return None

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
        body = self._build_body(messages, system, max_tokens, temperature, top_p, prefill)
        body["stream"] = True
        url = f"{API_BASE}?beta=true" if self._is_oauth else API_BASE

        # Emit prefill as the first chunk so downstream sees complete text
        if prefill:
            yield StreamChunk(content=prefill)

        # Pre-flight: check token validity (refresh on 401 before streaming)
        client = self._get_http_client()
        headers = self._get_headers()

        async with client.stream("POST", url, headers=headers, json=body) as response:
            if response.status_code == 401 and self._is_oauth:
                pass  # Fall through to retry below
            elif response.status_code != 200:
                text = await response.aread()
                raise RuntimeError(
                    f"Anthropic API error {response.status_code}: {text.decode()}"
                )
            else:
                # Success: stream normally
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:]
                    if payload.strip() == "[DONE]":
                        break
                    try:
                        event = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    event_type = event.get("type", "")
                    if event_type == "content_block_delta":
                        delta = event.get("delta", {})
                        if delta.get("type") == "text_delta":
                            yield StreamChunk(content=delta.get("text", ""))
                    elif event_type == "message_stop":
                        break
                yield StreamChunk(content="", is_final=True)
                return

        # 401 retry path: refresh token and stream again
        new_token = self._refresh_from_keychain()
        if new_token and new_token != self._api_key:
            print("[claude] stream token expired, refreshed from keychain")
            self.update_credentials(new_token)
            headers = self._get_headers()
            client = self._get_http_client()

            async with client.stream("POST", url, headers=headers, json=body) as response:
                if response.status_code != 200:
                    text = await response.aread()
                    raise RuntimeError(
                        f"Anthropic API error {response.status_code}: {text.decode()}"
                    )
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:]
                    if payload.strip() == "[DONE]":
                        break
                    try:
                        event = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    event_type = event.get("type", "")
                    if event_type == "content_block_delta":
                        delta = event.get("delta", {})
                        if delta.get("type") == "text_delta":
                            yield StreamChunk(content=delta.get("text", ""))
                    elif event_type == "message_stop":
                        break
            yield StreamChunk(content="", is_final=True)
        else:
            raise RuntimeError(
                "Anthropic API error 401: Token expired and could not refresh from keychain"
            )
