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
    fact_limit: int = 30,
) -> dict:
    """Snapshot live state into a dict ready for yaml.safe_dump."""
    data_dir = Path(data_dir)
    history = _read_history(data_dir, recipient_key, turns)
    clock = at or (history[-1][2] if history else datetime.now())

    store = FactStore(data_dir / "facts.db")
    facts: list[dict] = []
    for subject in ("aria", f"user:{recipient_key}"):
        for fact in await store.query(subject=subject, limit=fact_limit):
            age = clock - fact.ts
            facts.append({
                "subject": fact.subject,
                "type": fact.type.value,
                "source": fact.source.value,
                "content": fact.content,
                "importance": fact.importance,
                "days_ago": round(age.total_seconds() / 86400, 2),
            })

    history_rows = [
        {"role": role, "content": content,
         "minutes_ago": round((clock - ts).total_seconds() / 60, 1)}
        for role, content, ts in history[:-1]
    ]
    last_input = history[-1][1] if history else ""

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
