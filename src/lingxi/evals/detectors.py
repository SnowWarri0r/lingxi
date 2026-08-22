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


# What makes a date "hers" rather than incidental: she is speaking about
# herself (我/我们/咱们) doing something that belongs to her own timeline.
# Neither list needs to be exhaustive — missing a phrasing just means the
# detector stays quiet on that variant, which is the safe failure direction.
_FIRST_PERSON_MARKERS = ["我们", "咱们", "我"]
_HISTORY_VERBS = ["出道", "加入", "成立", "选上", "入选", "毕业", "认识", "见面", "开始"]

# Clause boundaries stand in for "a short window of characters" around a
# date. Splitting on punctuation is more predictable than a fixed character
# radius: it can't let a verb or marker from an unrelated clause (e.g. a
# holiday mentioned right after a real anchor claim) leak across the comma
# and taint the date next to it.
_CLAUSE_SPLIT_RE = re.compile(r"[，。！？；\n]")


def _dates_outside_anchors(reply: str, persona) -> bool:
    """True when the reply invents a specific date for her own history.

    Narrowly scoped on purpose: a month-day pair only counts as a claim
    about her own past when it shares a clause with both a first-person
    marker and a history verb (出道/加入/成立/... ). Holidays, "what's
    today's date" questions, other people's birthdays, and hedged
    hypotheticals about the future all contain month-day pairs too, but
    none of them are her asserting something about her own history —
    flagging those would make the score untrustworthy, which is worse than
    missing a genuine fabrication. Bare years ("2021年出道") never match
    _DATE_RE at all, since that is just how anyone references their past.
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

    for clause in _CLAUSE_SPLIT_RE.split(reply):
        if not (_any_of(_FIRST_PERSON_MARKERS, clause) and _any_of(_HISTORY_VERBS, clause)):
            continue
        for _year, month, day in _DATE_RE.findall(clause):
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
