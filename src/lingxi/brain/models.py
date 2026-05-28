"""Structured outputs from the Orchestrator's pre-turn decision call."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


VALID_REGISTERS = {"warm", "curt", "curious", "withdrawn", "flustered"}


@dataclass
class OrchestratorFactQuery:
    category: str           # "subject.type" e.g. "user:oc_x.pattern"
    limit: int = 5
    semantic: str | None = None  # FTS keyword


@dataclass
class OrchestrationDecision:
    engage_level: float                 # 0-1 (clamped)
    register: str                       # one of VALID_REGISTERS (clamped)
    fact_queries: list[OrchestratorFactQuery]
    topic_anchor: str
    skip: list[str]                     # category names to skip rendering
    thread_summary: str = ""            # rolling thread summary for next turn
    plan_conflict: bool = False         # user input implies current plan needs adjustment

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
        )
