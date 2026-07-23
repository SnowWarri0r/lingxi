"""Pre-turn web lookup — grounds the persona in an external fact she doesn't
carry in her own memory.

The orchestrator (Claude) decides a turn needs a lookup and emits a query;
this runs a web_search via the shared LLM provider (same server tool + auth
the daily news briefing uses) and returns a short factual grounding string.
The result is injected into the prompt BEFORE the responder speaks — the
responder (doubao) stays a single pass with no chat-time tools, so the voice
design is untouched. Fully fail-safe: any error / empty result returns "" and
the turn proceeds ungrounded.
"""

from __future__ import annotations

_LOOKUP_PROMPT = """用 web_search 查证下面这个问题，回**确凿的信息**：

问题：{query}

要求：
- 直接给事实；一条一行
- 若问的是台词/歌词/原话，就把**查到的原文照实引出来**（能引多少引多少，原文保持原语言，可附中文说明）
- 查到的照实给；查不到确切的就明说"没查到确切的"——**绝不编造原话或细节**
- 末尾用一行列出信息来源域名（如 来源：xxx.fandom.com）
"""


async def web_lookup(
    llm,
    query: str,
    *,
    allowed_domains: list[str] | None = None,
    max_searches: int = 3,
    max_tokens: int = 800,
) -> str:
    """Return a short factual grounding for `query`, or "" on any failure."""
    query = (query or "").strip()
    if not query:
        return ""

    tool: dict = {
        "type": "web_search_20250305",
        "name": "web_search",
        "max_uses": max_searches,
    }
    if allowed_domains:
        tool["allowed_domains"] = allowed_domains

    try:
        result = await llm.complete(
            messages=[{"role": "user", "content": _LOOKUP_PROMPT.format(query=query)}],
            max_tokens=max_tokens,
            tools=[tool],
            _debug_purpose="chat_web_lookup",
        )
    except Exception as e:
        print(f"[retrieval] web_lookup failed for {query!r}: {e}", flush=True)
        return ""

    text = (result.content or "").strip()
    if not text:
        return ""
    return text
