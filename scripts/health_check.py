"""Tell the person running the bot when it is down, once, and when it is back.

    python scripts/health_check.py

Run every few minutes by a launchd agent. The bot is kept alive by launchd,
but a crash on startup just repeats: on 10-27 a stale credential kept it
down, and it was noticed only because someone happened to read the log.

Down means any of: launchd has no process for it; its log has not been
written for STALE (the opener scheduler logs every five minutes, so silence
means a hang); or the log ends in a traceback. Alerts go to the Mac's
notification centre, not through the bot itself, which is the thing that
may be broken. State is kept so a long outage notifies once.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

LABEL = os.environ.get("LINGXI_LAUNCHD_LABEL", "com.lingxi.feishu")
LOG = Path(os.environ.get("LINGXI_LOG",
                          Path.home() / "Library" / "Logs" / "lingxi-feishu.log"))
STATE = Path(os.environ.get("LINGXI_HEALTH_STATE",
                            Path.home() / "Library" / "Logs" / "lingxi-health.json"))
STALE = 20 * 60          # seconds without a log line
TAIL = 4000              # bytes of log read for a traceback


def running(label: str) -> bool:
    r = subprocess.run(["launchctl", "list", label], capture_output=True, text=True)
    return r.returncode == 0 and '"PID"' in r.stdout


def tail(path: Path, n: int = TAIL) -> str:
    with open(path, "rb") as f:
        f.seek(0, 2)
        f.seek(max(0, f.tell() - n))
        return f.read().decode("utf-8", "replace")


def problem(*, alive: bool, log: Path, now: float) -> str:
    """Why the bot is down, or "" when it looks fine."""
    if not alive:
        return "进程没在跑"
    if not log.exists():
        return "找不到日志"
    if now - log.stat().st_mtime > STALE:
        return f"日志 {int((now - log.stat().st_mtime) // 60)} 分钟没更新"
    text = tail(log)
    last_tb = text.rfind("Traceback (most recent call last)")
    last_ok = max(text.rfind("connected"), text.rfind("[proactive]"))
    if last_tb > last_ok:
        lines = [l for l in text[last_tb:].splitlines() if l.strip()]
        return f"启动报错：{lines[-1][:80]}" if lines else "启动报错"
    return ""


def notify(title: str, body: str) -> None:
    script = f"display notification {json.dumps(body)} with title {json.dumps(title)}"
    subprocess.run(["osascript", "-e", script], capture_output=True)


def main() -> int:
    why = problem(alive=running(LABEL), log=LOG, now=time.time())
    try:
        was = json.loads(STATE.read_text()).get("down", "")
    except (OSError, ValueError):
        was = ""
    if why and not was:
        notify("lingxi 挂了", why)
    elif was and not why:
        notify("lingxi 恢复了", f"之前：{was}")
    if why != was:
        STATE.write_text(json.dumps({"down": why, "at": time.time()},
                                    ensure_ascii=False))
    print(f"[health] {why or 'ok'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
