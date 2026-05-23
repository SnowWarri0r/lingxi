"""Inspect captured LLM requests (see lingxi.debug.request_log).

Usage:
  # List today's requests (timestamp, purpose, model, token usage)
  uv run tools/inspect_llm.py list

  # Dump full record by request_id (or shorthand prefix)
  uv run tools/inspect_llm.py show <request_id>

  # Tail latest N, full bodies
  uv run tools/inspect_llm.py tail [N=1]

  # Filter by purpose: chat_full / think / compress / unknown / ...
  uv run tools/inspect_llm.py list --purpose chat_full

  # Pick a different day's log
  uv run tools/inspect_llm.py list --date 2026-05-19

Logs are at $MEMORY_DATA_DIR/../debug/llm_requests/YYYY-MM-DD.jsonl
(default ./data/debug/llm_requests/).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path


def log_dir() -> Path:
    base = os.environ.get("MEMORY_DATA_DIR", "./data/memory")
    return Path(base).expanduser().resolve().parent / "debug" / "llm_requests"


def load_day(date: str | None) -> list[dict]:
    date = date or datetime.now().strftime("%Y-%m-%d")
    path = log_dir() / f"{date}.jsonl"
    if not path.exists():
        return []
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def cmd_list(args) -> int:
    recs = load_day(args.date)
    if args.purpose:
        recs = [r for r in recs if r.get("purpose") == args.purpose]
    if not recs:
        print(f"(no records for {args.date or 'today'})")
        return 0
    for r in recs:
        ts = r.get("ts", "")[:23]
        rid = r.get("request_id", "?")
        purpose = r.get("purpose", "?")
        model = r.get("model", "?")
        u = r.get("usage", {})
        intok = u.get("input_tokens", 0)
        outtok = u.get("output_tokens", 0)
        dur = r.get("duration_ms", 0)
        resp = (r.get("response", "") or "").replace("\n", " ")[:80]
        print(f"{ts}  {rid}  {purpose:14s}  {model[:30]:30s}  {intok:>6}in/{outtok:>4}out  {dur:>5}ms  | {resp}")
    print(f"\n({len(recs)} records)")
    return 0


def cmd_show(args) -> int:
    recs = load_day(args.date)
    matches = [r for r in recs if r.get("request_id", "").startswith(args.id)]
    if not matches:
        print(f"no record matching '{args.id}'")
        return 1
    if len(matches) > 1:
        print(f"ambiguous: {len(matches)} matches, showing first")
    r = matches[0]
    print(f"=== ts:        {r.get('ts')}")
    print(f"=== id:        {r.get('request_id')}")
    print(f"=== purpose:   {r.get('purpose')}")
    print(f"=== model:     {r.get('model')}")
    print(f"=== duration:  {r.get('duration_ms')}ms")
    print(f"=== usage:     {r.get('usage')}")
    print()
    print("=== SYSTEM ===")
    print(r.get("system", "(empty)"))
    print()
    print("=== MESSAGES ===")
    for i, m in enumerate(r.get("messages", []), 1):
        role = m.get("role", "?")
        content = m.get("content", "")
        if isinstance(content, list):
            content = json.dumps(content, ensure_ascii=False, indent=2)
        print(f"--- [{i}] {role} ---")
        print(content)
    print()
    print("=== RESPONSE ===")
    print(r.get("response", ""))
    return 0


def cmd_tail(args) -> int:
    recs = load_day(args.date)
    if args.purpose:
        recs = [r for r in recs if r.get("purpose") == args.purpose]
    if not recs:
        print("(no records)")
        return 0
    n = max(1, args.n)
    for r in recs[-n:]:
        args.id = r["request_id"]
        cmd_show(args)
        print()
        print("=" * 80)
        print()
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--date", help="YYYY-MM-DD (default today)")
    p.add_argument("--purpose", help="filter by purpose (chat_full/think/compress/...)")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="one-line summary of each request")
    show = sub.add_parser("show", help="dump one record fully")
    show.add_argument("id", help="request_id prefix")
    tail = sub.add_parser("tail", help="show last N records fully")
    tail.add_argument("n", type=int, nargs="?", default=1)
    args = p.parse_args()
    if args.cmd == "list":
        return cmd_list(args)
    if args.cmd == "show":
        return cmd_show(args)
    if args.cmd == "tail":
        return cmd_tail(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
