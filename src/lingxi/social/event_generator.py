"""LLM-based generator for NPC events.

Called by the scheduler on each tick (only for NPCs that pass the dice
roll). Given the NPC's persona + recent events + active arcs, returns 1-2
new events with significance scores.

Conservative by design:
- Output is JSON list — on parse failure, returns empty (no event written)
- LLM is told most events should be daily-noise (significance 0.1-0.3)
- ~10% of events are aria_interaction (Aria + NPC do something together)
- Major events (significance ≥ 0.6) happen rarely
"""

from __future__ import annotations

import json
import random
import re
from datetime import datetime, timedelta
from typing import Any

from lingxi.providers.base import LLMProvider
from lingxi.social.models import NPC, NPCArc, NPCEvent, NPCState


_GENERATION_PROMPT = """你在帮 {npc_name}（{relation}）的生活里发生几件小事。

【这个人是谁】
{background}
性格：{traits}
跟 Aria 的相处：{interaction_style}

【最近 7 天发生过的事】
{recent_events}

【当前正在经历的 arc】（独立故事线，可能贯穿数周）
{active_arcs}

【当前时间】{now}（{time_of_day}）

请生成 1-2 个**真的可能在这个时间点发生**的小事。

要求：
- 大部分时候是日常小事（吃饭/通勤/学习/小情绪/朋友圈刷到什么），少数时候推进 arc
- 不要每个事件都跟 arc 相关，普通生活水占大多数
- 大约 1/10 概率是跟 Aria 一起经历的事，标 type=aria_interaction
- 给每个事件评 significance（0.0-1.0）：
  * 0.1-0.3：纯日常水（小敏点了 11 点的外卖）
  * 0.4-0.5：值得知道但不必转述（小敏导师 1on1 让她有点崩）
  * 0.6-0.8：重大事件（吵架/突破/坏消息）
  * 0.9+：极少（亲人病危/重大成就），不要轻易给
- arc_id 只在事件确实是这个 arc 推进时填，否则 null
- content 要具体（"小敏点了麦当劳的炸鸡"，不是"小敏吃东西了"）

输出严格 JSON 数组（不要 markdown 包裹，不要解释）：
[
  {{"type": "life", "content": "...", "significance": 0.x, "arc_id": null}},
  {{"type": "aria_interaction", "content": "拉 Aria 一起 ...", "significance": 0.x, "arc_id": "thesis_pressure"}}
]

如果这个时间点 {npc_name} 没什么值得记的事发生，返回 []。
"""


def _render_recent_events(events: list[NPCEvent], now: datetime) -> str:
    if not events:
        return "（最近没什么记下来的事）"
    cutoff = now - timedelta(days=7)
    recent = [e for e in events if e.ts >= cutoff][-10:]
    if not recent:
        return "（最近 7 天没什么记下来的事）"
    lines = []
    for e in recent:
        ts_str = e.ts.strftime("%m-%d %H:%M")
        tag = "[和 Aria]" if e.type == "aria_interaction" else ""
        lines.append(f"- {ts_str} {tag} {e.content}")
    return "\n".join(lines)


def _render_active_arcs(arcs: list[NPCArc]) -> str:
    active = [a for a in arcs if a.stage != "resolved"]
    if not active:
        return "（暂无正在经历的事）"
    lines = []
    for a in active:
        lines.append(f"- {a.summary}（stage={a.stage}, weight={a.weight}）")
    return "\n".join(lines)


def _time_of_day(now: datetime) -> str:
    h = now.hour
    if h < 6:
        return "凌晨"
    if h < 11:
        return "上午"
    if h < 14:
        return "中午"
    if h < 18:
        return "下午"
    if h < 22:
        return "晚上"
    return "深夜"


def _strip_json_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


async def generate_events(
    llm: LLMProvider,
    npc: NPC,
    state: NPCState,
    *,
    now: datetime | None = None,
    model: str | None = None,
) -> list[NPCEvent]:
    """Generate 0-2 new events for one NPC. Returns parsed NPCEvents.

    Empty list on LLM/parse failure — caller treats that as "nothing
    happened this tick". Never raises.
    """
    now = now or datetime.now()

    prompt = _GENERATION_PROMPT.format(
        npc_name=npc.name,
        relation=npc.relation,
        background=npc.background.strip(),
        traits="、".join(npc.traits) if npc.traits else "（未指定）",
        interaction_style=npc.interaction_style.strip() or "（未指定）",
        recent_events=_render_recent_events(state.recent_events, now),
        active_arcs=_render_active_arcs(state.arcs),
        now=now.strftime("%Y-%m-%d %H:%M"),
        time_of_day=_time_of_day(now),
    )

    try:
        kwargs: dict = {}
        if model:
            kwargs["model"] = model
        response = await llm.complete(
            messages=[{"role": "user", "content": prompt}],
            system="你在为一个虚构角色的生活生成日常小事。具体、克制、不戏剧化。",
            max_tokens=600,
            temperature=0.9,
            **kwargs,
        )
        text = response.content if hasattr(response, "content") else str(response)
        text = _strip_json_fences(text)
        data = json.loads(text)
    except Exception as e:
        print(f"[social.gen] LLM/parse failed for {npc.id}: {e}", flush=True)
        return []

    if not isinstance(data, list):
        return []

    out: list[NPCEvent] = []
    # Spread events slightly so they don't all share the exact same timestamp
    for i, raw in enumerate(data[:2]):
        if not isinstance(raw, dict):
            continue
        try:
            content = str(raw.get("content") or "").strip()
            if not content:
                continue
            etype = raw.get("type", "life")
            if etype not in ("life", "aria_interaction"):
                etype = "life"
            sig = float(raw.get("significance", 0.3))
            sig = max(0.0, min(1.0, sig))
            arc_id = raw.get("arc_id")
            if arc_id is not None:
                arc_id = str(arc_id).strip() or None
                # Drop arc_id if it doesn't match any of NPC's known arcs
                if arc_id and not any(a.id == arc_id for a in state.arcs):
                    arc_id = None
            ev_ts = now - timedelta(seconds=i)  # spread by 1s per event
            out.append(NPCEvent(
                npc_id=npc.id,
                ts=ev_ts,
                type=etype,
                content=content,
                significance=sig,
                arc_id=arc_id,
            ))
        except Exception:
            continue
    return out


def compute_tick_probability(
    npc: NPC,
    state: NPCState,
    *,
    now: datetime | None = None,
) -> float:
    """Probability this NPC should generate events on the current tick.

    Pure function, no LLM call — used by scheduler to roll the dice
    before paying for generation.

    Modulators:
    - base = npc.base_event_probability
    - +0.2 if no event in last 24h (catch-up)
    - +0.2 if any active arc has weight ≥ 0.7 (active storyline)
    - *0.3 if it's the 22h tick (winding down for the day)
    """
    now = now or datetime.now()
    p = npc.base_event_probability

    last = state.last_event_at
    if last is None or (now - last) > timedelta(hours=24):
        p += 0.2

    if any(a.weight >= 0.7 for a in state.active_arcs()):
        p += 0.2

    if now.hour == 22:
        p *= 0.3

    return max(0.0, min(1.0, p))


def should_tick(
    probability: float, *, rng: random.Random | None = None
) -> bool:
    """Roll the dice. Separate function so tests can inject deterministic rng."""
    rng = rng or random
    return rng.random() < probability
