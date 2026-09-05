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

# The single source of truth for which detector keys exist. `evaluate` and
# `Case`'s load-time validator both read this set, so a new detector only
# needs to be added here once — a second hardcoded list would drift.
KNOWN_DETECTORS = frozenset({"any_of", "regex", "regex_absent"})


def _any_of(needles: list[str], reply: str) -> bool:
    return any(n in reply for n in needles)


def _regex(pattern: str, reply: str) -> bool:
    return re.search(pattern, reply) is not None


def _regex_absent(pattern: str, reply: str) -> bool:
    """Fires when the reply does NOT contain `pattern`.

    For failures defined by something missing rather than something said.
    Enumerating the ways a fabricated memory can be phrased is hopeless — a
    first attempt at this case matched 3 of 20 replies that were all wrong,
    because she narrated a concert she had not played in twenty different
    vocabularies. What every correct reply must contain is small and closed:
    a statement that the date is still ahead.

    Only use this where the case's input demands that statement. Applied to
    an open-ended turn it marks every ordinary reply as a failure.
    """
    return re.search(pattern, reply) is None


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
    if "regex_absent" in spec:
        return _regex_absent(spec["regex_absent"], reply)
    raise ValueError(f"unknown detector: {sorted(spec)}")
