"""Daily news fetcher using Anthropic's web_search tool.

Bypasses our generic LLMProvider abstraction because it needs tool
support (multi-turn tool_use loop) that our streaming-focused provider
doesn't expose. The fetcher is offline / batch-only, so direct SDK use
is fine — it's not in the chat hot path.

The model's job:
1. Search a few queries (天文 / 文学 / 上海 / 科技 / 全球大事)
2. Pick 0-1 item per category, total ≤ 5
3. Re-voice each in Aria's IM register

Failure modes (network / quota / model refusal) return an empty
briefing so the chat path never breaks; logs the cause.
"""

from __future__ import annotations

import json
import re
from datetime import date, datetime
from typing import Any

from lingxi.world.models import DailyBriefing, NewsItem


_FETCH_PROMPT = """今天是 {today}。请用 web_search 查一下今天/昨天的新闻，从这些类目里挑：
- 天文 / 太空（NASA, JWST, 天文事件等）
- 文学 / 出版（新书, 文坛事件, 重要写作奖项）
- 上海本地（天气, 大事件, 文化活动）
- 科技 / AI（重要发布, 行业动态, 但不是无聊宣传稿）
- 全球大事（值得关注的国际新闻）

挑选标准：
- 每个类目 0-1 条**真正值得读到的**事，不必凑数
- 总数 **≤ 5 条**
- 跳过广告 / 无聊宣传稿 / 标题党
- 发生在最近 24-48 小时内

然后，用 Aria 的语气改写每一条——她是 28 岁的天文人 + 写作者，住上海，
contemplative 内向但好奇心强。她**不**用新闻播报口吻——她是"今早扫到的"那种
个人语气：简短、带一点自己的反应、IM 风格短句。

❌ "NASA 今日宣布在火星样本中发现微生物迹象"（新闻稿口吻）
✅ "今早扫到 NASA 说火星样本里有点微生物的迹象 真的吗"

❌ "上海今日预计降水概率 80%"
✅ "今天上海要下大雨"

输出严格 JSON（不要 markdown 包裹）：
{{
  "items": [
    {{
      "headline": "原标题或主题（<= 30 字）",
      "aria_voice": "她语气的一句（<= 50 字）",
      "category": "天文|文学|上海本地|科技|全球大事|其他",
      "source": "来源域名或媒体名",
      "url": "可选"
    }}
  ]
}}

如果今天实在没什么值得记的，items 给空 list 就行——比凑数好。"""


def _strip_json_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _extract_text_from_blocks(content_blocks: list) -> str:
    """The model may interleave tool_use blocks and text blocks. Final
    response text is in the last 'text' block (after all tool roundtrips
    have been resolved by the SDK)."""
    pieces: list[str] = []
    for block in content_blocks:
        block_type = getattr(block, "type", None) or (
            block.get("type") if isinstance(block, dict) else None
        )
        if block_type == "text":
            text = getattr(block, "text", None) or (
                block.get("text") if isinstance(block, dict) else ""
            )
            if text:
                pieces.append(text)
    return "\n".join(pieces)


async def fetch_daily_briefing(
    api_key: str,
    target_date: date | None = None,
    *,
    model: str = "claude-sonnet-4-5",
    max_tokens: int = 4000,
    max_searches: int = 5,
) -> DailyBriefing:
    """Fetch today's briefing using Claude + web_search tool.

    Returns an empty briefing on any failure (parse error, network
    timeout, tool unavailable). The caller (scheduler / chat path)
    treats empty == "no briefing today" gracefully.
    """
    if target_date is None:
        target_date = date.today()

    try:
        from anthropic import AsyncAnthropic
    except ImportError:
        print("[world] anthropic SDK not installed", flush=True)
        return DailyBriefing(date=target_date)

    client = AsyncAnthropic(api_key=api_key)

    prompt = _FETCH_PROMPT.format(today=target_date.isoformat())

    try:
        response = await client.messages.create(
            model=model,
            max_tokens=max_tokens,
            tools=[
                {
                    "type": "web_search_20250305",
                    "name": "web_search",
                    "max_uses": max_searches,
                }
            ],
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as e:
        print(f"[world] fetch API call failed: {e}", flush=True)
        return DailyBriefing(date=target_date)

    text = _extract_text_from_blocks(response.content)
    if not text:
        print("[world] fetch returned no text content", flush=True)
        return DailyBriefing(date=target_date)

    cleaned = _strip_json_fences(text)
    try:
        data: Any = json.loads(cleaned)
    except json.JSONDecodeError as e:
        print(f"[world] fetch JSON parse failed: {e}; raw[:200]={cleaned[:200]!r}", flush=True)
        return DailyBriefing(date=target_date)

    items_raw = data.get("items") if isinstance(data, dict) else None
    if not isinstance(items_raw, list):
        return DailyBriefing(date=target_date)

    items: list[NewsItem] = []
    valid_categories = {"天文", "文学", "上海本地", "科技", "全球大事", "其他"}
    now = datetime.now()
    for raw in items_raw:
        if not isinstance(raw, dict):
            continue
        headline = (raw.get("headline") or "").strip()
        aria_voice = (raw.get("aria_voice") or "").strip()
        if not headline or not aria_voice:
            continue
        category = raw.get("category", "其他")
        if category not in valid_categories:
            category = "其他"
        items.append(
            NewsItem(
                headline=headline[:80],
                aria_voice=aria_voice[:200],
                category=category,
                source=(raw.get("source") or "").strip()[:60],
                url=(raw.get("url") or "").strip()[:300],
                fetched_at=now,
            )
        )

    briefing = DailyBriefing(date=target_date, items=items, generated_at=now)
    print(
        f"[world] fetched {len(items)} items for {target_date.isoformat()}",
        flush=True,
    )
    return briefing
