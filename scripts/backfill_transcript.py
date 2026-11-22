"""Fill a Feishu recipient's transcript with what was said before it existed.

The transcript began with the live log; everything older was gone from the
30-turn buffer. Feishu keeps the chat, so his side can be read back:

    set -a && . ./.env && set +a
    uv run python scripts/backfill_transcript.py \
        --data data/personas/<slug> --chat <chat_id> [--utc-offset-before ...]

Only his messages come back as text. Her replies went out as streamed cards,
and the history API returns a card as a placeholder, so her side before the
live log is left out instead of guessed from debug logs that hold drafts
never sent. The turns still in the buffer are added with both sides.

Rows land before the earliest row already in the transcript, so running it
twice, or after the live log has started, adds nothing twice. Rows from here
carry a "source": feishu_history, debug_log (his messages as the
orchestrator logged them, for QQ and other stretches the history API does
not hold, with --with-debug-log), or buffer; live rows carry none.

Timestamps elsewhere in the data are naive local time of the process that
wrote them. When the process ran on another offset earlier, pass
`--utc-offset-before HOURS@ISO_UTC` so those messages read on the clock the
facts of that time were dated by.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lingxi.memory.short_term import ShortTermMemory  # noqa: E402
from lingxi.memory.transcript import TranscriptLog, _safe  # noqa: E402

BASE = "https://open.feishu.cn/open-apis"


def _token() -> str:
    r = httpx.post(f"{BASE}/auth/v3/tenant_access_token/internal", json={
        "app_id": os.environ["FEISHU_APP_ID"],
        "app_secret": os.environ["FEISHU_APP_SECRET"]}).json()
    if r.get("code") != 0:
        raise SystemExit(f"auth failed: {r}")
    return r["tenant_access_token"]


def text_of(item: dict) -> str:
    kind = item.get("msg_type")
    try:
        body = json.loads(item["body"]["content"])
    except (KeyError, ValueError, TypeError):
        return ""
    if kind == "text":
        return str(body.get("text") or "").strip()
    if kind == "post":
        blocks = body.get("content") or []
        return "\n".join("".join(seg.get("text", "") for seg in line if isinstance(seg, dict))
                         for line in blocks).strip()
    return {"image": "[图片]", "sticker": "[表情]", "file": "[文件]",
            "audio": "[语音]", "media": "[视频]"}.get(kind, "")


def his_messages(chat: str):
    headers = {"Authorization": f"Bearer {_token()}"}
    token = ""
    while True:
        params = {"container_id_type": "chat", "container_id": chat,
                  "page_size": 50, "sort_type": "ByCreateTimeAsc"}
        if token:
            params["page_token"] = token
        r = httpx.get(f"{BASE}/im/v1/messages", params=params, headers=headers).json()
        if r.get("code") != 0:
            raise SystemExit(f"list failed: {r}")
        data = r.get("data") or {}
        for it in data.get("items") or []:
            if (it.get("sender") or {}).get("sender_type") == "user" and not it.get("deleted"):
                text = text_of(it)
                if text:
                    yield int(it["create_time"]), text
        if not data.get("has_more"):
            return
        token = data.get("page_token") or ""


def debug_log_messages(data: str):
    """His messages as the orchestrator saw them, from the request log.

    Covers channels with no history API (QQ). Timestamps there are already
    the process's naive local time.
    """
    import glob
    import re
    pat = re.compile(r"【用户刚发的这一条】\n(.*?)\n\n?【", re.S)
    for p in sorted(glob.glob(str(Path(data) / "debug" / "llm_requests" / "*.jsonl"))):
        for raw in open(p, encoding="utf-8"):
            try:
                r = json.loads(raw)
            except ValueError:
                continue
            if r.get("purpose") != "orchestrator":
                continue
            msgs = r.get("messages") or []
            c = msgs[-1].get("content") if msgs else ""
            m = pat.search(c) if isinstance(c, str) else None
            if m and m.group(1).strip():
                yield datetime.fromisoformat(r["ts"]), m.group(1).strip()


def local_naive(ms: int, before: tuple[int, datetime] | None) -> datetime:
    utc = datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
    if before and utc < before[1]:
        return (utc + timedelta(hours=before[0])).replace(tzinfo=None)
    return utc.astimezone().replace(tzinfo=None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="the persona's data directory")
    ap.add_argument("--chat", required=True, help="Feishu chat_id")
    ap.add_argument("--utc-offset-before", default="",
                    help="HOURS@ISO_UTC: offset for messages before that moment")
    ap.add_argument("--since", default="",
                    help="YYYY-MM-DD: the persona's first day on this chat; a "
                         "chat reused across personas holds what was said to the others")
    ap.add_argument("--with-debug-log", action="store_true",
                    help="also take his messages from the request log, for "
                         "stretches the Feishu history does not hold")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    before = None
    if args.utc_offset_before:
        h, at = args.utc_offset_before.split("@", 1)
        before = (int(h), datetime.fromisoformat(at).replace(tzinfo=timezone.utc))

    key = f"feishu:{args.chat}"
    log = TranscriptLog(args.data)
    held = log.read(key)
    live = [r for r in held if "source" not in r]
    first_live = live[0]["ts"] if live else None

    import asyncio
    buffer = asyncio.run(ShortTermMemory(data_dir=args.data).snapshot_for_recipient(key))
    buf_start = buffer[0].timestamp if buffer else None

    rows = []
    for ms, text in his_messages(args.chat):
        ts = local_naive(ms, before)
        if args.since and ts < datetime.fromisoformat(args.since):
            continue
        # The buffer has these with her side too; his merged bursts there
        # would not match message by message anyway.
        if buf_start is not None and ts >= buf_start - timedelta(minutes=10):
            continue
        rows.append({"ts": ts, "role": "user", "content": text,
                     "source": "feishu_history"})
    if args.with_debug_log:
        have = [(r["ts"], r["content"]) for r in rows]
        for ts, text in debug_log_messages(args.data):
            if args.since and ts < datetime.fromisoformat(args.since):
                continue
            if buf_start is not None and ts >= buf_start - timedelta(minutes=10):
                continue
            # A burst he sent is one message to the orchestrator but several
            # in the chat history, so a match is any of his lines inside it.
            if any(abs((ts - t).total_seconds()) < 600 and c in text for t, c in have):
                continue
            rows.append({"ts": ts, "role": "user", "content": text,
                         "source": "debug_log"})
    rows += [{"ts": t.timestamp, "role": t.role, "content": t.content,
              "source": "buffer"} for t in buffer]
    if first_live is not None:
        rows = [r for r in rows if r["ts"] < first_live]
    done = {(r["ts"], r["content"]) for r in held}
    rows = [r for r in rows if (r["ts"], r["content"]) not in done]
    rows.sort(key=lambda r: r["ts"])

    print(f"{len(rows)} rows to add "
          f"({sum(r['source'] == 'feishu_history' for r in rows)} from history, "
          f"{sum(r['source'] == 'debug_log' for r in rows)} from the request log, "
          f"{sum(r['source'] == 'buffer' for r in rows)} from the buffer); "
          f"transcript already holds {len(held)}", flush=True)
    months = {}
    for r in rows:
        months[f"{r['ts']:%Y-%m}"] = months.get(f"{r['ts']:%Y-%m}", 0) + 1
    print("by month:", months, flush=True)
    if args.dry_run or not rows:
        return
    d = Path(args.data) / "transcripts" / _safe(key)
    d.mkdir(parents=True, exist_ok=True)
    for r in rows:
        line = json.dumps({**r, "ts": r["ts"].isoformat()}, ensure_ascii=False)
        with open(d / f"{r['ts']:%Y-%m}.jsonl", "a", encoding="utf-8") as f:
            f.write(line + "\n")
    print(f"written under {d}", flush=True)


if __name__ == "__main__":
    main()
