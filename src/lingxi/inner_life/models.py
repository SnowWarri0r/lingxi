"""Data models for Aria's inner life.

Everything here is AUTHORITATIVE STATE, not prose for the LLM to elaborate.
The LLM reads this as fact and responds as a person with this state.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime
from enum import Enum

from pydantic import BaseModel, Field


class ActivityKind(str, Enum):
    """Coarse categories of what she might be doing."""

    SLEEP = "sleep"
    ROUTINE = "routine"       # coffee, breakfast, shower, commute
    WORK = "work"             # writing, observing, research
    SOCIAL = "social"         # chatting with friend, call
    REST = "rest"             # lounging, reading for pleasure
    OUTDOORS = "outdoors"     # walk, stargazing, errands
    MEAL = "meal"
    HOBBY = "hobby"


class Activity(BaseModel):
    """A single activity in her day."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    kind: ActivityKind
    name: str              # short label, e.g., "改稿"
    description: str       # "在改关于仙女座那段，卡在开头一句"
    started_at: datetime
    ended_at: datetime | None = None
    # How this activity influences her conversation style/availability
    focus_level: float = Field(default=0.5, ge=0.0, le=1.0)  # high = harder to interrupt
    social_openness: float = Field(default=0.5, ge=0.0, le=1.0)  # high = eager to chat
    # Physical grounding — where she physically is during this activity.
    # Examples: "沙发", "书桌前", "厨房", "便利店", "床上". Empty = no scene.
    scene: str = ""


class LifeEvent(BaseModel):
    """Something that happened to her — not a planned activity."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    timestamp: datetime = Field(default_factory=datetime.now)
    content: str                              # "编辑回信说第三章要重写"
    emotional_impact: dict[str, float] = Field(default_factory=dict)  # {dim: delta}
    significance: float = Field(default=0.5, ge=0.0, le=1.0)
    wants_to_share: bool = False              # does she want to tell the user?


class DiaryEntry(BaseModel):
    """An internal narrative beat — her own voice, her own day."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    timestamp: datetime = Field(default_factory=datetime.now)
    content: str                              # "今天早上咖啡机坏了，只能去楼下买..."
    tags: list[str] = Field(default_factory=list)


class DailyPlan(BaseModel):
    """A generated-at-dawn sketch of how today goes."""

    date: date
    mood_theme: str = ""                      # "心情平静的一天" "有点焦虑"
    scheduled_activities: list[Activity] = Field(default_factory=list)
    pending_events: list[LifeEvent] = Field(default_factory=list)  # may trigger later
    generated_at: datetime = Field(default_factory=datetime.now)


class InnerState(BaseModel):
    """Aria's current moment-to-moment state. Persisted and updated by simulator."""

    current_activity: Activity | None = None
    # Rolling window of recent events (last 48h)
    recent_events: list[LifeEvent] = Field(default_factory=list)
    # Rolling diary (last 7 days of entries)
    recent_diary: list[DiaryEntry] = Field(default_factory=list)
    # Today's plan
    today_plan: DailyPlan | None = None
    # Energy and creative drive (beyond emotion dimensions)
    energy: float = Field(default=0.7, ge=0.0, le=1.0)
    creative_drive: float = Field(default=0.5, ge=0.0, le=1.0)
    social_need: float = Field(default=0.4, ge=0.0, le=1.0)  # wants to talk to people
    last_simulated_at: datetime | None = None
    # Ambient grounding (set at dawn, decays/refreshes over the day)
    sleep_quality: float = Field(default=0.7, ge=0.0, le=1.0)  # last night
    # Cap on per-day significant events (so days don't feel theme-saturated)
    significant_events_today: int = 0
    significant_events_reset_date: date | None = None
    # Reactive: when the user actually chats, social_need drops + we mark this
    last_chat_at: datetime | None = None


# ---------------------------------------------------------------------------
# Subjective layer — per-recipient impressions
# ---------------------------------------------------------------------------


class Impression(BaseModel):
    """A single adjectival impression of a person."""

    content: str                                      # "温暖但常常过度工作"
    created_at: datetime = Field(default_factory=datetime.now)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    last_reinforced: datetime = Field(default_factory=datetime.now)


class SubjectiveView(BaseModel):
    """How Aria subjectively sees one specific person."""

    recipient_key: str                                # "feishu:oc_xxx"
    impressions: list[Impression] = Field(default_factory=list)
    worries: list[str] = Field(default_factory=list)  # "他最近熬夜太多"
    appreciations: list[str] = Field(default_factory=list)  # "他认真"
    current_stance: str = "friendly-distant"          # free-form shorthand
    relationship_feeling: str = ""                    # her paragraph of how she feels about them
    last_updated: datetime = Field(default_factory=datetime.now)


# ---------------------------------------------------------------------------
# Agenda — things she wants to say/ask
# ---------------------------------------------------------------------------


class AgendaKind(str, Enum):
    SHARE = "share"               # "今天看到X，想告诉你"
    FOLLOW_UP = "follow_up"       # "上次你说Y，现在怎样了"
    CONCERN = "concern"           # "他熬夜太多了，想提醒一下"
    QUESTION = "question"         # "我突然想到一个问题想问他"
    INVITATION = "invitation"     # "想约他做什么"


class AgendaItem(BaseModel):
    """Something Aria wants to bring up with a specific recipient."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    recipient_key: str
    kind: AgendaKind
    content: str                                      # the gist of what she wants to say
    priority: float = Field(default=0.5, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=datetime.now)
    expires_at: datetime | None = None                # after this, less relevant
    delivered: bool = False                           # has she said it yet?
    delivered_at: datetime | None = None
    source: str = ""                                  # "life_event:xxx" or "reflection"
