"""Every turn, kept as it was said, for as long as the data directory lives.

The short-term buffer holds 30 turns and evicts the rest, and the facts are
what a model made of a conversation. When a fact turned out wrong there was
nothing left to check it against: the words it came from had aged out of
the buffer, and only scattered debug logs remained. This log is that ground
truth. It is append-only, one JSON line per turn, one file per month under
`transcripts/<recipient>/`.

The first write for a recipient starts from the turns the buffer still
holds, so a log begun on a live conversation does not open mid-sentence.
Writing never raises: losing a line of the record must not lose a reply.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


def _safe(key: str) -> str:
    return "".join(c if c.isalnum() or c in "-_:" else "_" for c in key)


class TranscriptLog:
    def __init__(self, data_dir: Path | str):
        self._root = Path(data_dir) / "transcripts"

    def _dir(self, key: str) -> Path:
        return self._root / _safe(key)

    def append(self, key: str, turn, seed=()) -> None:
        """Record `turn`; on a recipient's first write, `seed` goes first."""
        try:
            d = self._dir(key)
            rows = [] if d.exists() else [t for t in seed if t is not turn]
            d.mkdir(parents=True, exist_ok=True)
            for t in [*rows, turn]:
                ts = t.timestamp
                line = json.dumps({"ts": ts.isoformat(), "role": t.role,
                                   "content": t.content}, ensure_ascii=False)
                with open(d / f"{ts:%Y-%m}.jsonl", "a", encoding="utf-8") as f:
                    f.write(line + "\n")
        except Exception as e:
            print(f"[transcript] write failed (non-fatal): {e!r}", flush=True)

    def read(self, key: str, since: datetime | None = None,
             until: datetime | None = None) -> list[dict]:
        """Turns in time order, `since` inclusive, `until` exclusive."""
        d = self._dir(key)
        if not d.exists():
            return []
        out = []
        for p in sorted(d.glob("*.jsonl")):
            for raw in p.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(raw)
                    ts = datetime.fromisoformat(row["ts"])
                except (ValueError, KeyError, TypeError):
                    continue
                if (since is None or ts >= since) and (until is None or ts < until):
                    out.append({**row, "ts": ts})
        out.sort(key=lambda r: r["ts"])
        return out
