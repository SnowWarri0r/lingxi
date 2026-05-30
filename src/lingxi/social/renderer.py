"""Render the "身边的人" block for Aria's system prompt.

Pull-mode rendering: all NPCs are background knowledge in the system
prompt. Push (significance≥0.6 events) is handled separately by the
promoter writing into inner_state.recent_events.

Rendering strategy:
- Show up to MAX_NPCS NPCs per render to keep prompt budget bounded
- Prioritize NPCs with: high-weight active arc, recent event in last 48h
- Per NPC: name, relation, current arc summary (1 line), 1-2 recent events
"""

from __future__ import annotations

from datetime import datetime, timedelta

from lingxi.social.models import NPC, NPCEvent, NPCState, SocialGraph


MAX_NPCS = 4                # how many NPCs to render per turn
MAX_RECENT_EVENTS_PER = 2   # events shown per NPC
RECENT_WINDOW_HOURS = 72    # events older than this are "background only"


def render_social_section(
    graph: SocialGraph,
    states: dict[str, NPCState],
    *,
    now: datetime | None = None,
) -> str | None:
    """Build "身边的人" markdown block. Returns None when nothing to show."""
    now = now or datetime.now()
    cutoff = now - timedelta(hours=RECENT_WINDOW_HOURS)

    ranked = _rank_npcs(graph, states, cutoff)
    if not ranked:
        return None

    rendered: list[str] = []
    for npc in ranked[:MAX_NPCS]:
        state = states.get(npc.id)
        if state is None:
            continue
        block = _render_one_npc(npc, state, cutoff)
        if block:
            rendered.append(block)

    if not rendered:
        return None

    header = "【身边的人】（你生活里在意的人，最近他们身上发生的事）"
    return header + "\n\n" + "\n\n".join(rendered)


def _rank_npcs(
    graph: SocialGraph,
    states: dict[str, NPCState],
    cutoff: datetime,
) -> list[NPC]:
    """Sort NPCs by relevance — recent activity + arc weight."""

    def score(npc: NPC) -> float:
        state = states.get(npc.id)
        if state is None:
            return 0.0
        # Recent events score
        recent = [e for e in state.recent_events if e.ts >= cutoff]
        recency = sum(0.3 + e.significance for e in recent[-3:])
        # Active arc weight
        arc_weight = max(
            (a.weight for a in state.active_arcs()), default=0.0
        )
        return recency + arc_weight

    return sorted(graph.npcs, key=score, reverse=True)


def _render_one_npc(
    npc: NPC, state: NPCState, cutoff: datetime
) -> str | None:
    """Render one NPC block. Skip NPCs with no signal at all."""
    active_arcs = state.active_arcs()
    recent = [e for e in state.recent_events if e.ts >= cutoff]

    if not active_arcs and not recent:
        return None

    age_str = f"{npc.age}，" if npc.age is not None else ""
    # Collapse "妈妈 妈妈" → "妈妈" when name == relation (family terms)
    title = npc.relation if npc.name == npc.relation else f"{npc.relation} {npc.name}"
    head = f"**{title}**（{age_str}{_one_line_background(npc)}）"

    lines = [head]

    if active_arcs:
        # Show top-2 by weight
        arcs_sorted = sorted(active_arcs, key=lambda a: a.weight, reverse=True)[:2]
        for arc in arcs_sorted:
            lines.append(f"- 正在经历：{arc.summary}（{_stage_zh(arc.stage)}）")

    if recent:
        # Most recent first
        recent_sorted = sorted(recent, key=lambda e: e.ts, reverse=True)
        for ev in recent_sorted[:MAX_RECENT_EVENTS_PER]:
            tag = "和你" if ev.type == "aria_interaction" else ""
            ts_str = _short_ts(ev.ts)
            lines.append(f"- 最近{ts_str}{tag}：{ev.content}")

    return "\n".join(lines)


def _one_line_background(npc: NPC) -> str:
    """First non-empty line of background, trimmed."""
    for line in npc.background.splitlines():
        line = line.strip()
        if line:
            return line[:60]
    return npc.relation


def _stage_zh(stage: str) -> str:
    return {
        "early": "刚开始",
        "mid": "进行中",
        "climax": "正卡在节点上",
        "resolved": "已经过去了",
    }.get(stage, stage)


def _short_ts(ts: datetime) -> str:
    """Render timestamp as 今早/昨晚/前天 etc."""
    now = datetime.now()
    delta = now - ts
    if delta < timedelta(hours=12):
        h = ts.hour
        if 5 <= h < 12:
            return "今早"
        if 12 <= h < 18:
            return "今天下午"
        return "今晚"
    if delta < timedelta(days=1):
        return "昨天"
    if delta < timedelta(days=2):
        return "前天"
    return f"{delta.days}天前"


def collect_aria_interactions(
    states: dict[str, NPCState], *, since: datetime
) -> list[NPCEvent]:
    """Cross-NPC view of events where Aria was directly involved.

    Used at bootstrap to backfill Aria's recent_events from the NPC log
    after restart — so she "remembers" interactions that happened while
    she was offline.
    """
    out: list[NPCEvent] = []
    for state in states.values():
        for ev in state.recent_events:
            if ev.type == "aria_interaction" and ev.ts >= since:
                out.append(ev)
    out.sort(key=lambda e: e.ts)
    return out
