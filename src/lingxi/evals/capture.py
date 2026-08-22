"""Freeze a live turn into a case skeleton.

Everything mechanical is done here — clock, fact ages, turn offsets — so
the only things left to write are what went wrong and what counts as wrong.
The Feishu annotation channel is the cautionary tale: built, wired, shipped,
and sitting at 493 recorded turns with 0 annotations, because it costs one
extra click. A case library that costs twenty minutes an entry dies the
same way.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from lingxi.facts.store import FactStore


async def capture(
    recipient_key: str,
    *,
    persona_path: str,
    data_dir: Path | str,
    turns: int = 8,
    at: datetime | None = None,
    fact_limit: int = 8,
) -> dict:
    """Snapshot live state into a dict ready for yaml.safe_dump."""
    data_dir = Path(data_dir)
    raw_rows = _read_history(data_dir, recipient_key, turns)
    clock = at or (raw_rows[-1][2] if raw_rows else datetime.now())

    store = FactStore(data_dir / "facts.db", clock=lambda: clock)
    await store.init()
    facts: list[dict] = []
    for subject in ("aria", f"user:{recipient_key}"):
        for fact in await store.query(subject=subject, limit=fact_limit):
            age = clock - fact.ts
            seconds = age.total_seconds()
            fact_row = {
                "subject": fact.subject,
                "type": fact.type.value,
                "source": fact.source.value,
                "content": fact.content,
                "importance": fact.importance,
            }
            # Facts under a day old keep minute precision; the round-trip
            # through days_ago rounded to 2 decimals loses ~14 minutes,
            # which erases a fact that is only minutes old.
            if seconds < 86400:
                fact_row["minutes_ago"] = round(seconds / 60, 2)
            else:
                fact_row["days_ago"] = round(seconds / 86400, 2)
            facts.append(fact_row)

    last_input, history_before = _split_last_user(raw_rows)
    history_rows = [
        {"role": role, "content": content,
         "minutes_ago": round((clock - ts).total_seconds() / 60, 1)}
        for role, content, ts in history_before
    ]

    return {
        "id": f"{clock:%Y%m%d}-rename-me",
        "symptom": "",
        "origin": f"captured from {recipient_key} at {clock.isoformat()}",
        "persona": persona_path,
        "recipient": recipient_key,
        "clock": clock.isoformat(),
        "facts": facts,
        "history": history_rows,
        "input": last_input,
        "samples": 20,
        "premise": {"prompt_contains": [], "prompt_lacks": []},
        "detect": {"fail": {"any_of": []}},
        "budget": {"max_fail_rate": 0.05},
    }


def _read_history(
    data_dir: Path, recipient_key: str, turns: int,
) -> list[tuple[str, str, datetime]]:
    path = data_dir / "short_term" / f"{recipient_key}.json"
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows = raw.get("turns", raw if isinstance(raw, list) else [])
    out: list[tuple[str, str, datetime]] = []
    for row in rows[-turns:]:
        ts = row.get("timestamp")
        out.append((
            row.get("role", "user"),
            row.get("content", ""),
            datetime.fromisoformat(ts) if ts else datetime.now(),
        ))
    return out


def _split_last_user(
    rows: list[tuple[str, str, datetime]],
) -> tuple[str, list[tuple[str, str, datetime]]]:
    """Split raw turns into the user's last message (`input`) and everything
    strictly before it (`history`).

    The short-term buffer appends a "user" row while a turn is being
    prepared and an "assistant" row right after the reply is generated, so
    at rest the last row is always the assistant's own line. Treating that
    as `input` would feed the model its own prior sentence as if the user
    had just said it. Anything after the last user turn — the reply this
    case exists to re-generate — is dropped so it cannot leak the answer
    into the case. A buffer with no user turn at all yields no input and
    no history rather than guessing.
    """
    for i in range(len(rows) - 1, -1, -1):
        if rows[i][0] == "user":
            return rows[i][1], rows[:i]
    return "", []
