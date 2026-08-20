"""Deterministic detectors. Pure functions, no IO, no model calls.

Deliberately narrow. A detector that fires on something it should not makes
the whole score untrustworthy, and an untrusted score is worse than no
score — nobody acts on it. Missing a variant is cheap by comparison: it
shows up as the same symptom again and gets a second detector.

A `dates_outside_anchors` detector (substring/regex heuristics for "she
invented a date for her own history") was tried and removed. Judging
whether a date is a fabricated claim about herself needs subject, tense
and negation, not substring matching: review rounds kept surfacing new
false-positive classes (holidays, asking today's date, a third party's
date, hedged hypotheticals, a correct claim poisoned by an unrelated
mention in the same reply, unpunctuated run-ons that defeat clause
splitting, possessives where 我 modifies someone else, denials). A
detector that keeps growing new false-positive classes cannot meet
判定器宁可漏报不可误报. Fabrication detection is deferred to the
LLM-judge phase; do not re-add this as a substring/regex detector.
"""

from __future__ import annotations

import re


def _any_of(needles: list[str], reply: str) -> bool:
    return any(n in reply for n in needles)


def _regex(pattern: str, reply: str) -> bool:
    return re.search(pattern, reply) is not None


def evaluate(spec: dict, reply: str, persona=None) -> bool:
    """True when this detector fires on `reply`.

    `persona` is unused by the current detectors but stays in the
    signature: later tasks already call `evaluate` with three arguments,
    and future detectors will need persona context (e.g. anchors).
    """
    if "any_of" in spec:
        return _any_of(spec["any_of"], reply)
    if "regex" in spec:
        return _regex(spec["regex"], reply)
    raise ValueError(f"unknown detector: {sorted(spec)}")
