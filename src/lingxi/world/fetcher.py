"""Daily news fetcher using Anthropic's web_search tool.

Bypasses our generic LLMProvider abstraction because it needs tool
support (multi-turn tool_use loop) that our streaming-focused provider
doesn't expose. The fetcher is offline / batch-only, so direct SDK use
is fine — it's not in the chat hot path.

The model's job:
1. Search the persona's own `world_interests` categories
2. Pick 0-1 item per category, total ≤ 5, skipping what she scanned recently
3. Re-voice each in her IM register, in the language she speaks

Failure modes (network / quota / model refusal) return an empty
briefing so the chat path never breaks; logs the cause.
"""

from __future__ import annotations

import json
import re
from datetime import date, datetime
from typing import Any

from lingxi.world.models import DailyBriefing, NewsItem


_FETCH_PROMPT = """今天是 {today}。{where}请用 web_search 查一下今天/昨天的新闻，从这些类目里挑：
{topics_block}
{already_block}
挑选标准：
- 每个类目 0-1 条**真正值得读到的**事，不必凑数
- 总数 **≤ 5 条**
- 跳过广告 / 无聊宣传稿 / 标题党
- 发生在最近 24-48 小时内

然后，用下面这个人的语气改写每一条：

{self_context}

她**不**用新闻播报口吻——她是"今早扫到的"那种个人语气：简短、带一点自己的
反应、IM 风格短句。

**用上面那段自我描述所用的语言写**，那是她平时说话的语言。新闻原文是什么
语言不影响这一点——查到外文的，理解完用她的语言讲出来。

示例只示范**语气的分量**，别从里面取题材——题材看上面的类目：

❌ "某机构今日宣布相关项目将于下月正式启动"（新闻稿口吻）
✅ "今早扫到他们下个月就要开始弄那个了 真的假的"

❌ "该地区今日预计降水概率 80%"
✅ "那边明天要下大雨"

整条输出就是那个 JSON 对象本身。
**字符串值内一律用中文「」做引号**，确保 JSON 可解析。
{{
  "items": [
    {{
      "headline": "原标题或主题（<= 30 字）",
      "voice": "她语气的一句（<= 50 字）",
      "category": "上面那几个类目之一，原样抄",
      "source": "来源域名或媒体名",
      "url": "可选"
    }}
  ]
}}

如果今天实在没什么值得记的，items 给空 list 就行——比凑数好。"""


_ALREADY_TMPL = """
她这几天已经扫到过下面这些，**别再报一遍**——同一件事有真的新进展才提，
否则换别的：
{lines}
"""

# Three days of scanning is enough context to avoid a repeat; more is an
# archive that eats the prompt.
_MAX_ALREADY = 24


def build_fetch_prompt(persona, target_date: date,
                       recent: list[str] | None = None) -> str | None:
    """The search prompt for this persona, or None when she follows nothing.

    Both halves used to be literal text describing the first character this
    ran for — its categories and its writer. Any other persona then received
    that character's morning: a school idol woke up to telescope launches,
    re-voiced as a contemplative 28-year-old astronomer in Shanghai.
    """
    interests = [str(t).strip()
                 for t in (getattr(persona, "world_interests", None) or [])
                 if str(t).strip()]
    if not interests:
        return None
    from lingxi.persona.self_context import build_self_context
    # Which city counts as 外面. The weather block and the daylight calc both
    # read persona.location; the fetch did not, so on its first real day it
    # returned local weather for one country and the persona's home city as
    # somewhere else, in a prompt whose own weather line was the home city.
    from lingxi.temporal.sun import persona_location
    where = ""
    try:
        name = (persona_location(persona).name or "").strip()
        if name:
            where = (f"她人在{name}——「本地/外面」指的是{name}，"
                     f"别的地方要说清是哪儿。")
    except Exception:
        pass
    # What she already scanned. Without it the fetch re-reported the same
    # story on later days — 「8th的会场和日程出来了 有明→福冈→名古屋」 came back
    # twice in three days, both at importance 6, and the block pushes the one
    # highest-importance item, so that line was in her head on two of them.
    seen = [str(s).strip() for s in (recent or []) if str(s).strip()]
    already = (_ALREADY_TMPL.format(
        lines="\n".join(f"- {s}" for s in seen[:_MAX_ALREADY])) if seen else "")
    return _FETCH_PROMPT.format(
        today=target_date.isoformat(),
        where=where,
        topics_block="\n".join(f"- {t}" for t in interests),
        already_block=already,
        self_context=build_self_context(persona),
    )


def _strip_json_fences(text: str) -> str:
    text = text.strip()
    # web_search replies often wrap the JSON in a prose preamble + a ```json
    # fence ("Based on my searches… ```json{…}```"). Prefer the fenced block's
    # contents wherever it sits; fall back to the widest {...} span; else strip
    # leading/trailing fences as before.
    fenced = re.search(r"```(?:json)?\s*(.+?)\s*```", text, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    first, last = text.find("{"), text.rfind("}")
    if 0 <= first < last:
        return text[first:last + 1].strip()
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
    llm,
    persona,
    target_date: date | None = None,
    *,
    recent: list[str] | None = None,
    max_tokens: int = 4000,
    max_searches: int = 5,
) -> DailyBriefing:
    """Fetch today's briefing using the LLM provider + web_search tool.

    Routes through the shared ClaudeProvider so the call reuses whatever auth
    the bot runs on (OAuth Bearer or API key) — the OAuth path supports the
    web_search server tool, so no separate ANTHROPIC_API_KEY is needed.

    Returns an empty briefing on any failure (parse error, network timeout,
    tool unavailable). The caller treats empty == "no briefing today".
    """
    if target_date is None:
        target_date = date.today()

    prompt = build_fetch_prompt(persona, target_date, recent=recent)
    if prompt is None:
        # She follows nothing in particular; there is no morning to fetch.
        return DailyBriefing(date=target_date)

    # Retry once on empty/unparseable output: the model occasionally emits
    # invalid JSON (e.g. an unescaped " inside a value), which is random, so a
    # re-generation usually lands clean. Auth/network errors are NOT retried —
    # they won't fix themselves on a second call.
    data: Any = None
    for attempt in range(2):
        try:
            result = await llm.complete(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                tools=[
                    {
                        "type": "web_search_20250305",
                        "name": "web_search",
                        "max_uses": max_searches,
                    }
                ],
                _debug_purpose="world_fetch",
            )
        except Exception as e:
            print(f"[world] fetch API call failed: {e}", flush=True)
            return DailyBriefing(date=target_date)

        # result.content already concatenates text blocks (search-result blocks
        # are dropped); fall back to raw blocks if content is empty.
        text = result.content or _extract_text_from_blocks(
            getattr(result, "raw_content_blocks", []))
        if not text:
            print("[world] fetch returned no text content", flush=True)
            continue

        cleaned = _strip_json_fences(text)
        try:
            data = json.loads(cleaned)
            break
        except json.JSONDecodeError as e:
            print(f"[world] fetch JSON parse failed (attempt {attempt + 1}): "
                  f"{e}; raw[:200]={cleaned[:200]!r}", flush=True)
            data = None

    if data is None:
        return DailyBriefing(date=target_date)

    items_raw = data.get("items") if isinstance(data, dict) else None
    if not isinstance(items_raw, list):
        return DailyBriefing(date=target_date)

    items: list[NewsItem] = []
    now = datetime.now()
    for raw in items_raw:
        if not isinstance(raw, dict):
            continue
        headline = (raw.get("headline") or "").strip()
        # `aria_voice` is the old key; personas that are not Aria still get
        # read correctly if a cached prompt or an older model reply uses it.
        voice = (raw.get("voice") or raw.get("aria_voice") or "").strip()
        if not headline or not voice:
            continue
        # Categories are the persona's own interests now, so there is no
        # whitelist to check against — just a length bound.
        category = str(raw.get("category") or "其他").strip()[:40] or "其他"
        items.append(
            NewsItem(
                headline=headline[:80],
                voice=voice[:200],
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
