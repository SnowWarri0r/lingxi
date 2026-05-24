"""Schemas for Aria's social graph (NPCs in her life).

Three layers:
- NPC: persona definition (handwritten in yaml, immutable at runtime)
- NPCArc: an ongoing storyline (论文压力 / 相亲 / paper 等审稿), has stages
- NPCEvent: a single thing that happened (append-only log)

Arcs and events live in data/social/npcs/{npc_id}/ — yaml is config,
events.jsonl + arcs.json are runtime state.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


ArcStage = Literal["early", "mid", "climax", "resolved"]
EventType = Literal["life", "aria_interaction"]


class NPCArc(BaseModel):
    """An ongoing storyline in this NPC's life.

    Stages progress early → mid → climax → resolved. arc_advancer
    decides when to advance based on event_count and LLM judgment.
    Resolved arcs stop generating events but stay in the log for 30
    days before archive (so Aria can still reference them).
    """

    id: str
    summary: str
    stage: ArcStage = "early"
    weight: float = 0.5                    # 0-1, current prominence in NPC's life
    started_at: datetime = Field(default_factory=datetime.now)
    last_advanced_at: datetime = Field(default_factory=datetime.now)
    event_count: int = 0
    resolution: str | None = None          # filled when stage="resolved"


class NPCEvent(BaseModel):
    """A single thing that happened to (or with) this NPC.

    Significance drives the push/pull decision:
    - 0.1-0.3: pure daily noise (eating / commuting)
    - 0.4-0.5: noteworthy background
    - 0.6-0.8: major event — promoter pushes to Aria.recent_events
    - 0.9+: rare crisis / breakthrough
    """

    npc_id: str
    ts: datetime
    type: EventType
    content: str
    significance: float
    arc_id: str | None = None
    promoted_to_aria: bool = False         # idempotency flag for promoter


class NPC(BaseModel):
    """Static persona definition for one NPC. Loaded from yaml.

    Runtime state (arcs progressing, new events) lives separately in
    data/social/npcs/{id}/ — NOT mutated here.
    """

    id: str
    name: str
    relation: str                          # 室友 / 导师 / 妈妈 / 闺蜜 / ...
    age: int | None = None
    background: str                        # multi-line, who this person is
    traits: list[str] = Field(default_factory=list)
    interaction_style: str = ""            # how they interact with Aria specifically
    base_event_probability: float = 0.3    # per-tick base prob, modulated by recency/arc
    initial_arcs: list[NPCArc] = Field(default_factory=list)


class SocialGraph(BaseModel):
    """Top-level container for one persona's NPC roster."""

    npcs: list[NPC]

    def by_id(self, npc_id: str) -> NPC | None:
        for n in self.npcs:
            if n.id == npc_id:
                return n
        return None


class NPCState(BaseModel):
    """Runtime state for one NPC: arcs + recent events.

    Loaded by store from arcs.json + events.jsonl. Not persisted as a
    single blob — arcs/events have their own files for append-friendly
    writes.
    """

    npc_id: str
    arcs: list[NPCArc] = Field(default_factory=list)
    recent_events: list[NPCEvent] = Field(default_factory=list)
    last_event_at: datetime | None = None

    def active_arcs(self) -> list[NPCArc]:
        return [a for a in self.arcs if a.stage != "resolved"]
