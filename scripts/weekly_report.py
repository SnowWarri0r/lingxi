"""Week by week: how her openers landed and how her voice held.

    python scripts/weekly_report.py [persona_id] [--weeks N]

Read-only, prints to the terminal. These are the numbers that were being
pulled by hand every few days — reply rate, which kinds of opener went out,
how much of it was practice, her '!' and self-naming rates — so a change can
be judged across weeks rather than on one evening.

Weeks are reported, not compared: one person's week is a dozen openers, and a
rate moving between two weeks is mostly noise. Read the run of weeks.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

PRACTICE = "练习排练"


def _week(d: datetime) -> str:
    monday = (d - timedelta(days=d.weekday())).date()
    return monday.isoformat()


def opener_rows(records: list[dict], key: str) -> dict[str, dict]:
    """Per week: sent, answered, and kinds, from the outcomes kept with each
    opener."""
    rec = next((r for r in records
                if f"{r.get('channel')}:{r.get('recipient_id')}" == key), None)
    weeks: dict[str, dict] = defaultdict(lambda: {"sent": 0, "answered": 0,
                                                  "kinds": Counter()})
    for o in (rec or {}).get("proactive_outcomes", []):
        w = weeks[_week(datetime.fromisoformat(o["sent_at"]))]
        w["sent"] += 1
        w["answered"] += bool(o.get("answered"))
        w["kinds"][o.get("aspect") or "（未标）"] += 1
    return weeks


def voice_rows(turns: list[dict], self_name: str) -> dict[str, object]:
    from lingxi.persona.voice_monitor import measure
    by: dict[str, list[str]] = defaultdict(list)
    for t in turns:
        if t.get("role") == "assistant":
            by[_week(datetime.fromisoformat(t["timestamp"]))].append(t["content"])
    return {w: measure(xs, self_name) for w, xs in by.items()}


def main(argv: list[str]) -> int:
    weeks_n = 6
    if "--weeks" in argv:
        weeks_n = int(argv[argv.index("--weeks") + 1])
    pid = next((a for a in argv if not a.startswith("--") and not a.isdigit()),
               "tangkeke")
    data = REPO / "data" / "personas" / pid
    raw = json.loads((data / "interactions.json").read_text(encoding="utf-8"))
    records = raw if isinstance(raw, list) else list(raw.get("records", raw).values())
    name = ""
    try:
        # The name she calls herself by, as the live voice monitor counts it.
        from lingxi.persona.loader import load_persona
        persona = load_persona(REPO / "config" / "personas" / f"{pid}.yaml")
        name = str((getattr(persona, "voice_baseline", None) or {}).get("self_name") or "")
    except Exception:
        pass
    agenda = {}
    if (data / "agenda.json").exists():
        agenda = json.loads((data / "agenda.json").read_text(encoding="utf-8"))

    for rec in records:
        key = f"{rec.get('channel')}:{rec.get('recipient_id')}"
        ops = opener_rows(records, key)
        if not ops:
            continue
        buf = data / "short_term" / f"{key}.json"
        turns = json.loads(buf.read_text(encoding="utf-8"))["turns"] if buf.exists() else []
        # The buffer keeps her last 30 turns, so the voice columns reach back
        # only as far as those go; within a week they read her latest dozen,
        # as the live voice monitor does, and stay blank under its minimum.
        voice = voice_rows(turns, name)
        print(f"== {key[:14]}…")
        print(f"{'周一':10} {'主动':>4} {'回了':>4} {'练舞':>4}  {'!':>5} {'自称':>5}  话题")
        for w in sorted(set(ops) | set(voice))[-weeks_n:]:
            o = ops.get(w, {"sent": 0, "answered": 0, "kinds": Counter()})
            v = voice.get(w)
            kinds = "、".join(f"{k}{n}" for k, n in o["kinds"].most_common())
            print(f"{w:10} {o['sent']:>4} {o['answered']:>4} "
                  f"{o['kinds'].get(PRACTICE, 0):>4}  "
                  f"{(f'{v.exclaim:.2f}' if v else '—'):>5} "
                  f"{(f'{v.self_name:.2f}' if v else '—'):>5}  {kinds}")
        items = agenda.get(key, [])
        if items:
            st = Counter(i.get("status") for i in items)
            asked = sum(len(i.get("asked") or []) for i in items)
            print(f"议程：{dict(st)}，跟进过 {asked} 次")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
