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
from pydantic import BaseModel, ConfigDict, Field, field_validator

from lingxi.evals.detectors import KNOWN_DETECTORS
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


def _check_detector_spec(spec: dict, field_name: str) -> dict:
    """Reject a detector spec that names no known detector, or that names one
    correctly but can never fire.

    Both mistakes are indistinguishable from a working case until sampling
    finishes: an unedited `--capture` skeleton (`any_of: []`) reports PASS
    on every reply, and a typoed key either raises after 20 paid samples
    (unknown-key branch of `evaluate`) or, if it happens to collide with
    another known key, silently never fires. Catching this at case-load
    time, before anything is spent, is the whole point of validating here.
    """
    if not spec:
        raise ValueError(
            f"detect.{field_name} 是空的——如果这是刚 --capture 出来的骨架，"
            f"还需要手写 detect（例如 {{'any_of': ['命中的原句']}}）"
        )
    unknown = sorted(set(spec) - KNOWN_DETECTORS)
    if unknown:
        raise ValueError(
            f"detect.{field_name} 用了未知判定器 {unknown}；"
            f"已知判定器是 {sorted(KNOWN_DETECTORS)}"
        )
    if "any_of" in spec and not spec["any_of"]:
        raise ValueError(
            f"detect.{field_name}.any_of 是空列表，永远不会命中——"
            f"如果这是刚 --capture 出来的骨架，还需要手写命中的原句/关键词"
        )
    if "regex" in spec and not spec["regex"]:
        raise ValueError(
            f"detect.{field_name}.regex 是空字符串，永远不会命中，需要手写正则"
        )
    return spec


class Detect(_Strict):
    fail: dict
    passing: dict | None = Field(default=None, alias="pass")

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    @field_validator("fail")
    @classmethod
    def _check_fail(cls, v: dict) -> dict:
        return _check_detector_spec(v, "fail")

    @field_validator("passing")
    @classmethod
    def _check_passing(cls, v: dict | None) -> dict | None:
        if v is None:
            return v
        return _check_detector_spec(v, "pass")


class Acquaintance(_Strict):
    """How long the two have known each other, as of the frozen clock.

    Without this the replay wires no interaction tracker, and the per-turn
    reminder tells the model 「这是你们第一次对话」 while sitting above
    dozens of turns of history — a register shift applied to every case.
    """

    days: int = 0
    turns: int = 0
    relationship_level: int = 1


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
    acquaintance: Acquaintance = Field(default_factory=Acquaintance)

    def last_interaction(self) -> datetime | None:
        """When they last spoke before this turn — the newest frozen turn."""
        rows = self.resolved_history()
        return max((ts for _r, _c, ts in rows), default=None)

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
