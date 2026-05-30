"""Bidirectional NPC↔Aria interaction.

A single social event produces TWO facts — one from Aria's first-person
view, one from the NPC's first-person view. Two LLM calls (separate
voices) keeps each output cleanly in one persona.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from lingxi.facts.models import Fact, FactType, Source
from lingxi.facts.retriever import FactQuery, FactRetriever
from lingxi.facts.writers.life import LifeWriter
from lingxi.facts.writers.npc import NPCWriter
from lingxi.providers.base import LLMProvider


_ARIA_SYSTEM_TEMPLATE = "你是 Aria，刚才跟 {npc} 有了一次互动，现在用一句话记一下。"
_NPC_SYSTEM_TEMPLATE = "你是 {npc}，刚才跟 Aria 有了一次互动，现在用一句话记一下。"


_ARIA_PROMPT = """情境：{scenario}
{npc} 现在大概在：{npc_plan}
我今天大概在：{aria_plan}
我们最近的交集：
{shared_history}

记一下这次互动给我留下的印象——一两句，具体细节（对方说了什么 / 我的反应 / 一个感觉）。
"""

_NPC_PROMPT_TEMPLATE = """情境：{scenario}
Aria 现在大概在：{aria_plan}
我今天大概在：{npc_plan}
我们最近的交集：
{shared_history}

记一下这次互动给我留下的印象——一两句，具体细节。
"""


async def bidirectional_interaction(
    *,
    llm: LLMProvider,
    retriever: FactRetriever,
    life_writer: LifeWriter,
    npc_writer: NPCWriter,
    npc_id: str,
    npc_display: str,
    scenario: str,
    model: str | None = None,
) -> None:
    subject_npc = f"npc:{npc_id}"
    now = datetime.now()

    aria_plan = await _current_plan_summary(retriever, "aria", now)
    npc_plan = await _current_plan_summary(retriever, subject_npc, now)
    shared = await _shared_history(retriever, subject_npc, npc_id, now)
    shared_text = "\n".join(f"  - {f.content}" for f in shared) or "（最近没什么具体交集）"

    aria_prompt = _ARIA_PROMPT.format(
        scenario=scenario,
        npc=npc_display,
        npc_plan=npc_plan or "（不太清楚）",
        aria_plan=aria_plan or "（在做手头的事）",
        shared_history=shared_text,
    )
    aria_view = await _safe_complete(
        llm,
        _ARIA_SYSTEM_TEMPLATE.format(npc=npc_display),
        aria_prompt,
        purpose="interaction_aria_view",
        model=model,
    )
    if aria_view:
        try:
            await life_writer.write(Fact(
                subject="aria",
                content=aria_view,
                source=Source.LIFE_SIMULATED,
                type=FactType.EVENT,
                ts=now,
                tags=[f"interaction_with:{npc_id}"],
            ))
        except Exception as e:
            print(f"[interaction] aria-side write failed: {e}", flush=True)

    npc_prompt = _NPC_PROMPT_TEMPLATE.format(
        scenario=scenario,
        aria_plan=aria_plan or "（看上去在忙）",
        npc_plan=npc_plan or "（在做自己的事）",
        shared_history=shared_text,
    )
    npc_view = await _safe_complete(
        llm,
        _NPC_SYSTEM_TEMPLATE.format(npc=npc_display),
        npc_prompt,
        purpose="interaction_npc_view",
        model=model,
    )
    if npc_view:
        try:
            await npc_writer.write(Fact(
                subject=subject_npc,
                content=npc_view,
                source=Source.NPC_TICKER,
                type=FactType.EVENT,
                ts=now,
                tags=["interaction_with:aria"],
            ))
        except Exception as e:
            print(f"[interaction] npc-side write failed: {e}", flush=True)


async def _current_plan_summary(
    retriever: FactRetriever, subject: str, now: datetime
) -> str:
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    plans = await retriever._store.query(
        subject=subject, type=FactType.PLAN, since=today_start, limit=20
    )
    for plan in plans:
        for t in plan.tags:
            if not t.startswith("time_window:"):
                continue
            try:
                window = t.removeprefix("time_window:")
                start_s, end_s = window.split("-")
                start_h = int(start_s.split(":")[0])
                end_h = int(end_s.split(":")[0])
                if start_h <= now.hour < end_h:
                    return plan.content
            except (ValueError, IndexError):
                continue
    return ""


async def _shared_history(
    retriever: FactRetriever, subject_npc: str, npc_id: str, now: datetime
) -> list[Fact]:
    week_ago = now - timedelta(days=7)
    aria_about_npc = await retriever.fetch(FactQuery(
        subject="aria", semantic=npc_id,
        since=week_ago, limit=3,
    ))
    npc_recent = await retriever.fetch(FactQuery(
        subject=subject_npc, type=FactType.EVENT,
        since=week_ago, limit=3,
    ))
    return aria_about_npc + npc_recent


async def _safe_complete(
    llm: LLMProvider, system: str, prompt: str, *, purpose: str, model: str | None
) -> str:
    try:
        kwargs = {"model": model} if model else {}
        response = await llm.complete(
            messages=[{"role": "user", "content": prompt}],
            system=system,
            max_tokens=200,
            temperature=0.7,
            _debug_purpose=purpose,
            **kwargs,
        )
        return response.content.strip()
    except Exception as e:
        print(f"[interaction] {purpose} failed: {e}", flush=True)
        return ""
