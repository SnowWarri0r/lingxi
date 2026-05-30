"""Decide when an NPC arc should progress to its next stage.

Stages: early → mid → climax → resolved.

Called by the scheduler after each tick's event writes (whether or not
events generated). Triggers an LLM judgment only when the arc has
accumulated enough events for the current stage. Without this, arcs
either stagnate forever (no resolution) or jump stages randomly.

Hard safeguard: any arc with event_count > MAX_EVENTS_PER_ARC is
force-resolved to prevent indefinite accumulation.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any

from lingxi.providers.base import LLMProvider
from lingxi.social.models import NPC, NPCArc, NPCEvent, NPCState
from lingxi.social.store import SocialStore


# Per-stage minimum events before we consider advancing.
# Tuned to keep arcs feeling load-bearing: not too fast (one event won't
# flip the world) but not too slow (a thesis crisis can resolve in ~2
# weeks of daily events).
STAGE_THRESHOLDS = {
    "early": 3,
    "mid": 4,
    "climax": 1,
}

# Force-resolve any arc with more than this many events to avoid
# arcs that consume an NPC's whole life forever.
MAX_EVENTS_PER_ARC = 20


_ADVANCE_PROMPT = """你在判断一个角色身上正在经历的事（arc）是否应该推进到下一阶段。

【这个人是谁】
{npc_name}（{relation}）：{background_brief}

【这个 arc 当前状态】
- 一句话概括：{summary}
- 当前阶段：{stage}
- 已经累积了 {event_count} 件相关事件

【这个 arc 相关的事件（按时间从早到晚）】
{events}

阶段含义：
- early：事情刚冒头，主角还在适应/试探
- mid：进展中，主角在主动应对
- climax：到关键节点，要做决定/要见结果
- resolved：已经过去，得到了结局（好结局或坏结局）

判断：
1. 是否该推进到下一阶段？基于：事件密度够不够、事件的指向性（是不是真在推进，还是在原地打转）
2. 如果推进：新的 summary（一句话，要包含已经发生的变化）
3. 如果直接到 resolved：resolution 是什么（一句话说结局）

输出严格 JSON（不要 markdown 包裹）：
{{
  "advance": true|false,
  "new_summary": "..." or null,
  "resolution": "..." or null
}}
"""


def _next_stage(current: str) -> str | None:
    chain = ["early", "mid", "climax", "resolved"]
    if current not in chain:
        return None
    idx = chain.index(current)
    if idx == len(chain) - 1:
        return None
    return chain[idx + 1]


def _strip_json_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _render_arc_events(events: list[NPCEvent], arc_id: str, limit: int = 8) -> str:
    relevant = [e for e in events if e.arc_id == arc_id]
    if not relevant:
        return "（无）"
    shown = relevant[-limit:]
    lines = []
    for e in shown:
        ts = e.ts.strftime("%m-%d %H:%M")
        lines.append(f"- {ts} {e.content}")
    return "\n".join(lines)


async def maybe_advance_arc(
    llm: LLMProvider,
    npc: NPC,
    arc: NPCArc,
    state: NPCState,
    *,
    now: datetime | None = None,
    model: str | None = None,
) -> NPCArc | None:
    """Judge whether `arc` should advance. Returns updated arc or None.

    Returns None means no change. Returns a new NPCArc to replace the
    existing one in the store.

    Force-resolve safeguard: if event_count exceeds MAX_EVENTS_PER_ARC,
    arc is resolved without LLM call.
    """
    now = now or datetime.now()

    if arc.stage == "resolved":
        return None

    # Force-resolve guard
    if arc.event_count > MAX_EVENTS_PER_ARC:
        return arc.model_copy(update={
            "stage": "resolved",
            "weight": max(arc.weight - 0.3, 0.1),
            "last_advanced_at": now,
            "resolution": f"（自动收束）{arc.summary} 持续了太久，自然淡化。",
        })

    threshold = STAGE_THRESHOLDS.get(arc.stage, 99)
    if arc.event_count < threshold:
        return None

    background_brief = (npc.background or "").strip().split("\n", 1)[0][:80]
    prompt = _ADVANCE_PROMPT.format(
        npc_name=npc.name,
        relation=npc.relation,
        background_brief=background_brief,
        summary=arc.summary,
        stage=arc.stage,
        event_count=arc.event_count,
        events=_render_arc_events(state.recent_events, arc.id),
    )

    try:
        kwargs: dict = {}
        if model:
            kwargs["model"] = model
        response = await llm.complete(
            messages=[{"role": "user", "content": prompt}],
            system="你在判断一个虚构角色的人生事件是否进入下一阶段。诚实、克制、按事件支持的程度判断。",
            max_tokens=500,
            temperature=0.4,
            **kwargs,
        )
        text = response.content if hasattr(response, "content") else str(response)
        data = json.loads(_strip_json_fences(text))
    except Exception as e:
        print(f"[social.arc] LLM/parse failed for {npc.id}/{arc.id}: {e}", flush=True)
        return None

    if not isinstance(data, dict) or not data.get("advance"):
        return None

    nxt = _next_stage(arc.stage)
    if nxt is None:
        return None

    new_summary = (data.get("new_summary") or arc.summary).strip()
    resolution = data.get("resolution")
    if resolution is not None:
        resolution = str(resolution).strip() or None

    # If LLM said advance but the answer is "resolved", honor that even
    # if next-step chain would only go to "mid" — i.e. LLM is allowed to
    # skip stages by directly providing a resolution.
    if resolution:
        return arc.model_copy(update={
            "stage": "resolved",
            "summary": new_summary,
            "resolution": resolution,
            "weight": max(arc.weight - 0.2, 0.1),
            "last_advanced_at": now,
            # Reset event_count so post-resolution still works if arc
            # gets reopened (rare; this is just a safe default)
            "event_count": 0,
        })

    return arc.model_copy(update={
        "stage": nxt,
        "summary": new_summary,
        "weight": min(arc.weight + 0.1, 1.0) if nxt == "climax" else arc.weight,
        "last_advanced_at": now,
        "event_count": 0,
    })


async def advance_npc_arcs(
    llm: LLMProvider,
    npc: NPC,
    store: SocialStore,
    *,
    now: datetime | None = None,
    model: str | None = None,
) -> int:
    """Check all arcs for one NPC, advance the ones that should, persist.

    Returns the number of arcs that advanced (for logging).
    """
    state = await store.load_state(npc.id)
    if not state.arcs:
        return 0

    changed = False
    new_arcs: list[NPCArc] = []
    advanced_count = 0
    for arc in state.arcs:
        updated = await maybe_advance_arc(
            llm, npc, arc, state, now=now, model=model,
        )
        if updated is None:
            new_arcs.append(arc)
        else:
            new_arcs.append(updated)
            changed = True
            advanced_count += 1
            print(
                f"[social.arc] {npc.id}/{arc.id} {arc.stage} → {updated.stage} "
                f"({updated.summary[:40]}...)",
                flush=True,
            )

    if changed:
        await store.save_arcs(npc.id, new_arcs)
    return advanced_count
