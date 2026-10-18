"""Notices when her typing drifts from how she is meant to sound, and says so.

Her own earlier messages are the strongest style signal in the prompt. In
September that loop held a persona whose top trait is 元气 at zero '!' in 51
messages; fixing the persona text barely moved it (1-2 of 33 replays) because
the flat history outweighed it, and it took a one-off context reset to break.
Nothing would notice the next slide.

This reads her recent messages to one person, compares a few surface rates
with the band the persona declares (`voice_baseline` in its YAML), and when
one is out of band returns a line for the per-turn state: positive, in her
terms, naming what to return to. A persona that declares no band is never
checked. The reasoning lives here, not in the line — the line is about her.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# How many of her latest messages to judge by, and the fewest worth judging.
WINDOW = 12
MIN_MESSAGES = 8

_EXCLAIM = re.compile(r"[!！]")


@dataclass
class VoiceStats:
    n: int
    exclaim: float     # share of messages with at least one '!'
    shouting: float    # share with three or more
    self_name: float   # share that refer to herself by name

    def summary(self) -> str:
        return (f"n={self.n} exclaim={self.exclaim:.2f} "
                f"shouting={self.shouting:.2f} self_name={self.self_name:.2f}")


def measure(messages: list[str], self_name: str = "") -> VoiceStats | None:
    xs = [m for m in messages if (m or "").strip()][-WINDOW:]
    if len(xs) < MIN_MESSAGES:
        return None
    n = len(xs)
    return VoiceStats(
        n=n,
        exclaim=sum(bool(_EXCLAIM.search(x)) for x in xs) / n,
        shouting=sum(len(_EXCLAIM.findall(x)) >= 3 for x in xs) / n,
        self_name=(sum(self_name in x for x in xs) / n) if self_name else 0.0,
    )


def out_of_band(stats: VoiceStats | None, baseline: dict | None) -> bool:
    """Too flat or too loud — the drift that a context reset is for. Self-name
    alone does not count: one line moves that well enough."""
    if stats is None or not baseline:
        return False
    lo, hi = baseline.get("exclaim_min"), baseline.get("shouting_max")
    return ((lo is not None and stats.exclaim < lo)
            or (hi is not None and stats.shouting > hi))


# A reset once she has read out of band on this many consecutive days, and
# not again within this many days of the last one.
DRIFT_DAYS = 3
RESET_COOLDOWN_DAYS = 7


def note_day(days: list[str], today: str, drifting: bool) -> list[str]:
    """The run of consecutive drifting days ending today, updated."""
    if not drifting:
        return []
    if days and days[-1] == today:
        return days
    from datetime import date, timedelta
    if days and date.fromisoformat(days[-1]) != date.fromisoformat(today) - timedelta(days=1):
        days = []
    return (days + [today])[-DRIFT_DAYS:]


def correction(stats: VoiceStats | None, baseline: dict | None) -> str | None:
    """The line to put in front of her, or None while she sounds like herself."""
    if stats is None or not baseline:
        return None
    lines = []
    lo = baseline.get("exclaim_min")
    hi = baseline.get("shouting_max")
    if lo is not None and stats.exclaim < lo:
        lines.append(baseline.get("quiet_note")
                     or "这几天你打字比你平常收着——回到你平时那股劲。")
    elif hi is not None and stats.shouting > hi:
        lines.append(baseline.get("loud_note")
                     or "这几天你打字比你平常吵——回到你平时的音量。")
    name_lo = baseline.get("self_name_min")
    if name_lo is not None and stats.self_name < name_lo:
        note = baseline.get("self_name_note")
        if note:
            lines.append(note)
    if not lines:
        return None
    return "## 你这几天说话的样子\n" + "\n".join(f"- {l}" for l in lines)


def voice_block(persona, messages: list[str]) -> str | None:
    """Measure and, if she has drifted, the line to add. Logs the reading."""
    baseline = getattr(persona, "voice_baseline", None) or {}
    if not baseline:
        return None
    stats = measure(messages, str(baseline.get("self_name") or ""))
    block = correction(stats, baseline)
    if stats is not None:
        print(f"[voice] {stats.summary()}"
              f"{' → correcting' if block else ''}", flush=True)
    return block
