"""Fetch + parse douban group-topic pages (server-rendered HTML).

Deterministic regex parse (no bs4/lxml installed). fetch is an injectable
async callable returning (status, final_url, text) so tests run offline.
"""

from __future__ import annotations

import re
from typing import Awaitable, Callable

_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
_TOPIC_URL = "https://www.douban.com/group/topic/{tid}/"

_TITLE = re.compile(r"<title>(.*?)</title>", re.S)
_OP = re.compile(r'<div class="topic-richtext">(.*?)</div>', re.S)
_REPLY = re.compile(
    r'<div class="reply-content">\s*<div class="markdown">(.*?)</div>', re.S)
_TAGS = re.compile(r"<[^>]+>")


async def _httpx_fetch(url: str) -> tuple[int, str, str]:
    import httpx
    async with httpx.AsyncClient(timeout=20, follow_redirects=True,
                                 headers={"User-Agent": _UA}) as c:
        r = await c.get(url)
        return r.status_code, str(r.url), r.text


def _strip(html: str) -> str:
    return re.sub(r"\s+", " ", _TAGS.sub(" ", html)).strip()


def parse_topic(html: str) -> tuple[str, str, list[str]]:
    """Return (title, op_text, replies). Replies tag-stripped + deduped,
    order preserved."""
    tm = _TITLE.search(html)
    title = _strip(tm.group(1)) if tm else ""
    title = re.split(r"\s*[-|]\s*", title)[0].strip()
    om = _OP.search(html)
    op = _strip(om.group(1)) if om else ""
    seen: set[str] = set()
    replies: list[str] = []
    for block in _REPLY.findall(html):
        text = _strip(block)
        if text and text not in seen:
            seen.add(text)
            replies.append(text)
    return title, op, replies


async def fetch_topic(
    topic_id: str,
    *,
    fetch: Callable[[str], Awaitable[tuple[int, str, str]]] = _httpx_fetch,
) -> str | None:
    """Fetch a topic page. Returns HTML, or None if blocked (sec.douban
    redirect) or non-200 (dead/deleted thread)."""
    status, final_url, text = await fetch(_TOPIC_URL.format(tid=topic_id))
    if status != 200 or "sec.douban.com" in final_url:
        return None
    return text
