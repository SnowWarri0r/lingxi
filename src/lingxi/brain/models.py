"""Structured outputs from the Orchestrator's pre-turn decision call."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


VALID_REGISTERS = {"light", "warm", "curt", "curious", "withdrawn", "flustered"}


@dataclass
class OrchestratorFactQuery:
    category: str           # "subject.type" e.g. "user:oc_x.pattern"
    limit: int = 5
    semantic: str | None = None  # FTS keyword


_HEDGE_MARKERS = ("或", "可能", "也许", "大概", "不确定", "说不定", "?", "？")


def _decisive(state: str) -> str:
    """Keep user_state only when it commits to one situation.

    Measured on the failing turn, 20 samples per condition: a decisive state
    ("还在公司，想下班还没到点") put the responder wrong 1-2 times in 20, while
    a hedged one ("还在公司或刚下班途中") put it wrong 5 in 20 — worse than
    supplying no state at all, because the responder answers the later branch.
    Since a hedge underperforms silence, it is dropped and the clock fallback
    takes over. Filtering here rather than in the prompt: telling the model not
    to hedge still produced hedges in 1 run of 3.
    """
    return "" if any(m in state for m in _HEDGE_MARKERS) else state


@dataclass
class OrchestrationDecision:
    engage_level: float                 # 0-1 (clamped)
    register: str                       # one of VALID_REGISTERS (clamped)
    fact_queries: list[OrchestratorFactQuery]
    topic_anchor: str
    skip: list[str]                     # category names to skip rendering
    thread_summary: str = ""            # rolling thread summary for next turn
    plan_conflict: bool = False         # user input implies current plan needs adjustment
    # A concise web-search query when the turn needs an external fact the
    # persona wouldn't reliably know from her own memory (canon detail,
    # real-world fact, current event). Empty on ordinary chat. Retrieval runs
    # pre-turn and its result is injected as grounding — the responder stays
    # a single pass with no chat-time tools.
    lookup_query: str = ""
    # Durable facts about the interlocutor worth keeping past this turn
    # (schedule, people, projects, preferences, commitments). Extraction is an
    # analysis job, so it belongs here rather than as a side-task for the
    # responder, which is busy composing one line of speech and skips it.
    memory_writes: list[str] = field(default_factory=list)
    # Where the interlocutor is and what he's doing right now, as evidenced by
    # this conversation. The prompt has always carried 【你此刻】 for her but
    # nothing for him, so his situation was left to a clock table keyed to a
    # generic 9-to-5 — which had him home from work at 20:23 while he was
    # sitting at his desk saying 想下班了. This is read off the live turns, so
    # it outranks the clock. Empty when the conversation gives no evidence.
    user_state: str = ""

    @classmethod
    def default(cls) -> "OrchestrationDecision":
        return cls(
            engage_level=0.6,
            register="warm",
            fact_queries=[
                OrchestratorFactQuery(category="aria.event", limit=3),
            ],
            topic_anchor="",
            skip=[],
            thread_summary="",
            plan_conflict=False,
            lookup_query="",
            memory_writes=[],
        )

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "OrchestrationDecision":
        register = raw.get("register", "warm")
        if register not in VALID_REGISTERS:
            register = "warm"

        engage = float(raw.get("engage_level", 0.6))
        engage = max(0.0, min(1.0, engage))

        queries_raw = raw.get("fact_queries") or []
        queries: list[OrchestratorFactQuery] = []
        for q in queries_raw:
            if not isinstance(q, dict):
                continue
            cat = q.get("category")
            if not cat:
                continue
            queries.append(OrchestratorFactQuery(
                category=str(cat),
                limit=int(q.get("limit", 5)),
                semantic=q.get("semantic"),
            ))

        return cls(
            engage_level=engage,
            register=register,
            fact_queries=queries,
            topic_anchor=str(raw.get("topic_anchor", "")),
            skip=[str(s) for s in raw.get("skip", [])],
            thread_summary=str(raw.get("thread_summary", "")),
            plan_conflict=bool(raw.get("plan_conflict", False)),
            lookup_query=str(raw.get("lookup_query") or "").strip(),
            memory_writes=[
                s for s in (
                    str(m).strip() for m in (raw.get("memory_writes") or [])
                ) if s
            ],
            user_state=_decisive(str(raw.get("user_state") or "").strip()),
        )
