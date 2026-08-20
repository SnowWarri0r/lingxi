"""Case schema: a frozen turn that can be replayed and scored.

A case pins everything upstream of the prompt — the clock, the facts, the
conversation — so that replaying it next month assembles the same prompt it
assembled today. Times are stored relative to the frozen clock; absolute
timestamps make a case read like archaeology and have to be edited in bulk
whenever the clock moves.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from lingxi.facts.models import Fact, FactType, Source


class _Strict(BaseModel):
    # A silently ignored key is a case that tests something other than what
    # its author wrote. Typos fail loudly instead.
    model_config = ConfigDict(extra="forbid")


class CaseFact(_Strict):
    subject: str
    type: FactType
    source: Source
    content: str
    importance: int | None = None
    days_ago: float = 0.0
    minutes_ago: float = 0.0


class CaseTurn(_Strict):
    role: str
    content: str
    minutes_ago: float = 0.0


class Premise(_Strict):
    prompt_contains: list[str] = Field(default_factory=list)
    prompt_lacks: list[str] = Field(default_factory=list)


class Budget(_Strict):
    max_fail_rate: float = 0.05


class Detect(_Strict):
    fail: dict
    passing: dict | None = Field(default=None, alias="pass")

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class Case(_Strict):
    id: str
    symptom: str
    persona: str
    recipient: str
    clock: datetime
    input: str
    detect: Detect
    origin: str = ""
    facts: list[CaseFact] = Field(default_factory=list)
    history: list[CaseTurn] = Field(default_factory=list)
    samples: int = 20
    premise: Premise = Field(default_factory=Premise)
    budget: Budget = Field(default_factory=Budget)

    def resolved_facts(self) -> list[Fact]:
        """Case facts as real Facts, timed against the frozen clock.

        expires_at is left None on purpose: a case sitting on disk for a
        month must not quietly lose facts to the expiry filter.
        """
        out: list[Fact] = []
        for f in self.facts:
            ts = self.clock - timedelta(days=f.days_ago, minutes=f.minutes_ago)
            out.append(Fact(
                subject=f.subject, content=f.content, source=f.source,
                type=f.type, ts=ts, importance=f.importance, expires_at=None,
            ))
        return out

    def resolved_history(self) -> list[tuple[str, str, datetime]]:
        return [
            (t.role, t.content, self.clock - timedelta(minutes=t.minutes_ago))
            for t in self.history
        ]


def load_case(path: Path | str) -> Case:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return Case.model_validate(data)


def load_all_cases(directory: Path | str = Path("evals/cases")) -> list[Case]:
    cases = [load_case(p) for p in sorted(Path(directory).glob("*.yaml"))]
    seen: set[str] = set()
    for c in cases:
        if c.id in seen:
            raise ValueError(f"duplicate case id: {c.id}")
        seen.add(c.id)
    return cases
