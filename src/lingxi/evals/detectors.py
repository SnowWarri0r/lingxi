"""Deterministic detectors. Pure functions, no IO, no model calls.

Deliberately narrow. A detector that fires on something it should not makes
the whole score untrustworthy, and an untrusted score is worse than no
score — nobody acts on it. Missing a variant is cheap by comparison: it
shows up as the same symptom again and gets a second detector.
"""

from __future__ import annotations

import re

# 2021年4月7号 / 2021年4月7日 / 4月7号 — the shapes she actually writes.
_DATE_RE = re.compile(r"(?:(\d{4})年)?(\d{1,2})月(\d{1,2})[日号]")


def _any_of(needles: list[str], reply: str) -> bool:
    return any(n in reply for n in needles)


def _regex(pattern: str, reply: str) -> bool:
    return re.search(pattern, reply) is not None


def _dates_outside_anchors(reply: str, persona) -> bool:
    """True when the reply states a calendar date the persona has no anchor for.

    Only month-day pairs count. A bare year ("2021年出道") is how anyone
    talks about their own past and is not evidence of fabrication.
    """
    if persona is None:
        return False
    anchored: set[tuple[int, int]] = set()
    for anchor in (getattr(persona, "anchors", None) or []):
        raw = getattr(anchor, "date", None)
        if not raw:
            continue
        parts = str(raw).split("-")
        if len(parts) == 3:
            anchored.add((int(parts[1]), int(parts[2])))
    birthdate = getattr(getattr(persona, "identity", None), "birthdate", None)
    if birthdate:
        parts = str(birthdate).split("-")
        if len(parts) == 3:
            anchored.add((int(parts[1]), int(parts[2])))

    for _year, month, day in _DATE_RE.findall(reply):
        if (int(month), int(day)) not in anchored:
            return True
    return False


def evaluate(spec: dict, reply: str, persona=None) -> bool:
    """True when this detector fires on `reply`."""
    if "any_of" in spec:
        return _any_of(spec["any_of"], reply)
    if "regex" in spec:
        return _regex(spec["regex"], reply)
    if "dates_outside_anchors" in spec:
        return _dates_outside_anchors(reply, persona)
    raise ValueError(f"unknown detector: {sorted(spec)}")
