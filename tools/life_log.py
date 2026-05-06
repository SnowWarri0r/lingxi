"""View Aria's life trajectory — what she did and felt each day.

Reads:
  data/memory/inner_life/state.json       # current snapshot + recent_events
  data/memory/inner_life/diary/*.json     # one file per day
  data/memory/inner_life/biography_addenda.json  # reflection-grown bio

Usage:
  .venv/bin/python tools/life_log.py             # last 7 days + now
  .venv/bin/python tools/life_log.py --days 14   # last N days
  .venv/bin/python tools/life_log.py 2026-05-04  # single day deep view
  .venv/bin/python tools/life_log.py --now       # current state only
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path


DATA_DIR = Path("data/memory/inner_life")
DIARY_DIR = DATA_DIR / "diary"
STATE_PATH = DATA_DIR / "state.json"
ADDENDA_PATH = DATA_DIR / "biography_addenda.json"


def _load_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[!] failed to load {path}: {e}", file=sys.stderr)
        return None


def _fmt_dt(s: str) -> str:
    try:
        return datetime.fromisoformat(s).strftime("%H:%M")
    except Exception:
        return s[:16]


def _fmt_age(ts_str: str, now: datetime) -> str:
    try:
        ts = datetime.fromisoformat(ts_str)
    except Exception:
        return ""
    delta = now - ts
    if delta < timedelta(minutes=10):
        return "刚刚"
    if delta < timedelta(hours=1):
        return f"{int(delta.total_seconds()//60)}分钟前"
    if delta < timedelta(hours=24):
        return f"{int(delta.total_seconds()//3600)}小时前"
    if delta < timedelta(days=7):
        return f"{delta.days}天前"
    return ts.strftime("%m-%d")


def show_now():
    """Render the current state snapshot."""
    state = _load_json(STATE_PATH)
    if not state:
        print("（state.json 不存在或读取失败）")
        return

    print("─" * 60)
    print("📍 此刻")
    print("─" * 60)

    ca = state.get("current_activity") or {}
    if ca:
        scene = f" @ {ca.get('scene')}" if ca.get("scene") else ""
        started = _fmt_dt(ca.get("started_at", ""))
        ended = _fmt_dt(ca.get("ended_at", "")) if ca.get("ended_at") else "?"
        print(f"  正在: {ca.get('name')}{scene}")
        print(f"        {ca.get('description', '')}")
        print(f"        {started}-{ended}  专注 {ca.get('focus_level', 0):.1f}  社交开放 {ca.get('social_openness', 0):.1f}")

    plan = state.get("today_plan") or {}
    if plan:
        print(f"\n  今日基调: {plan.get('mood_theme', '')}")
        acts = plan.get("scheduled_activities") or []
        if acts:
            print(f"  日程 ({len(acts)} 段):")
            for a in acts:
                scene = f" @ {a.get('scene')}" if a.get("scene") else ""
                start = _fmt_dt(a.get("started_at", ""))
                end = _fmt_dt(a.get("ended_at", "")) if a.get("ended_at") else "?"
                print(f"    {start}-{end}  {a.get('name', '')}{scene}")
                if a.get("description"):
                    print(f"               {a.get('description')}")

    print(f"\n  状态  能量 {state.get('energy', 0):.2f}  创作 {state.get('creative_drive', 0):.2f}  "
          f"社交需求 {state.get('social_need', 0):.2f}")
    print(f"  睡眠质量 {state.get('sleep_quality', 0):.2f}  今日显著事件 {state.get('significant_events_today', 0)}/3")

    # Recent events
    events = state.get("recent_events") or []
    if events:
        now = datetime.now()
        print(f"\n  最近事件 ({len(events)} 条，按时间倒序):")
        for e in events[:12]:
            ts = e.get("timestamp", "")
            age = _fmt_age(ts, now)
            sig = e.get("significance", 0)
            share = "📌" if e.get("wants_to_share") else " "
            mark = "★" if sig >= 0.5 else "·"
            content = e.get("content", "")
            emo = e.get("emotional_impact") or {}
            emo_str = " ".join(f"{k}{v:+.1f}" for k, v in list(emo.items())[:2])
            print(f"    {share}{mark} [{age:>6}] {content}")
            if emo_str:
                print(f"            {emo_str}")


def show_day(date_str: str):
    """Render one day's diary entries in detail."""
    p = DIARY_DIR / f"{date_str}.json"
    if not p.exists():
        print(f"（{date_str} 没有 diary 文件）")
        return

    data = _load_json(p)
    if not data:
        return

    print("─" * 60)
    print(f"📔 {date_str}")
    print("─" * 60)

    entries = data.get("entries") or []
    if not entries:
        print("  （这一天没有日记条目）")
        return

    for e in entries:
        ts = _fmt_dt(e.get("timestamp", ""))
        tags = e.get("tags") or []
        tag_str = "/".join(tags) if tags else ""
        marker = {
            "daybreak": "🌅",
            "plan": "🌅",
            "event": "·",
            "quiet": "🌙",
        }.get(tags[0] if tags else "", "·")
        print(f"  {marker} {ts}  [{tag_str}]")
        print(f"          {e.get('content', '')}")


def show_recent_days(days: int):
    """Summary of the last N days."""
    today = datetime.now().date()
    print("─" * 60)
    print(f"📅 最近 {days} 天的轨迹")
    print("─" * 60)

    for offset in range(days - 1, -1, -1):
        d = today - timedelta(days=offset)
        ds = d.isoformat()
        p = DIARY_DIR / f"{ds}.json"
        if not p.exists():
            continue
        data = _load_json(p) or {}
        entries = data.get("entries") or []
        if not entries:
            continue

        weekday = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][d.weekday()]
        print(f"\n  {ds} ({weekday}) — {len(entries)} 条")

        for e in entries:
            ts = _fmt_dt(e.get("timestamp", ""))
            tags = e.get("tags") or []
            marker = {
                "daybreak": "🌅",
                "plan": "🌅",
                "event": " ·",
                "quiet": "🌙",
            }.get(tags[0] if tags else "", " ·")
            content = e.get("content", "")[:80]
            print(f"    {marker} {ts}  {content}")


def show_addenda():
    """Show biography events grown by reflection loop."""
    data = _load_json(ADDENDA_PATH)
    if not data:
        return
    if not isinstance(data, list):
        return

    print("─" * 60)
    print(f"📖 反思生长出来的传记事件 ({len(data)} 条)")
    print("─" * 60)
    for entry in data[-10:]:
        ev = entry.get("event") or {}
        created = entry.get("created_at", "")[:10]
        age = ev.get("age", "?")
        content = ev.get("content", "")
        tags = "/".join(ev.get("tags") or [])
        source = entry.get("source", "")
        print(f"  · [{created}] [{age}岁·{source}] {content}")
        if tags:
            print(f"          tags: {tags}")


def main() -> int:
    parser = argparse.ArgumentParser(description="View Aria's life trajectory")
    parser.add_argument("date", nargs="?", help="YYYY-MM-DD for single day deep view")
    parser.add_argument("--days", type=int, default=7, help="Last N days summary")
    parser.add_argument("--now", action="store_true", help="Current state only")
    parser.add_argument("--addenda", action="store_true", help="Show reflection-grown bio events")
    args = parser.parse_args()

    if args.now:
        show_now()
        return 0

    if args.date:
        show_day(args.date)
        return 0

    show_now()
    print()
    show_recent_days(args.days)
    print()
    if args.addenda or True:  # always show by default
        show_addenda()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
