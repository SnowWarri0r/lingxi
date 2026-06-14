"""Sense what the user is doing with Claude Code, so the pet can react.

Reads the most-recently-active Claude Code session log under
~/.claude/projects/<proj>/<session>.jsonl, tails it (logs reach 80MB+ — we
seek to the last chunk, never read the whole file), and classifies the
current activity from the last meaningful event:

  assistant + tool_use   → tool_running  (a tool is mid-flight)
  user + tool_result     → thinking      (tool finished, model is generating)
  assistant + text       → awaiting_user (turn done, waiting on the human)
  user (plain)           → thinking      (just sent a prompt)
  file mtime stale       → idle          (stepped away / nothing happening)
  no session file        → no_session

This is heuristic and intentionally cheap — it runs on a timer. The companion
layer turns transitions between these into in-character utterances.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path


# Seconds of no log activity after which we call it idle regardless of the
# last event kind (the human stepped away mid-turn or the agent is done).
IDLE_AFTER_SECS = 90.0


@dataclass
class ActivitySignal:
    kind: str          # tool_running | thinking | awaiting_user | idle | no_session
    detail: str        # short hint, e.g. "Bash: pytest -q" / "Edit: engine.py"
    idle_secs: float   # seconds since the session file was last modified
    session_id: str    # session file stem (stable id for the current run)

    @property
    def is_active(self) -> bool:
        return self.kind in ("tool_running", "thinking")


def _latest_session(root: Path) -> Path | None:
    """Most-recently-modified session jsonl across all projects."""
    if not root.exists():
        return None
    best: Path | None = None
    best_m = -1.0
    for proj in root.iterdir():
        if not proj.is_dir():
            continue
        for f in proj.glob("*.jsonl"):
            try:
                m = f.stat().st_mtime
            except OSError:
                continue
            if m > best_m:
                best_m, best = m, f
    return best


def _tail_lines(path: Path, max_bytes: int = 65536) -> list[str]:
    """Last lines of a (possibly huge) file without reading it all."""
    try:
        size = path.stat().st_size
        with path.open("rb") as fh:
            if size > max_bytes:
                fh.seek(size - max_bytes)
            data = fh.read()
    except OSError:
        return []
    text = data.decode("utf-8", "ignore")
    # First line after a mid-file seek may be a fragment — drop it.
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if size > max_bytes and lines:
        lines = lines[1:]
    return lines


def _tool_detail(content: list) -> str:
    for c in content:
        if isinstance(c, dict) and c.get("type") == "tool_use":
            name = c.get("name", "") or ""
            inp = c.get("input") or {}
            if name == "Bash":
                cmd = str(inp.get("command", "")).strip().replace("\n", " ")
                return f"Bash: {cmd[:48]}" if cmd else "Bash"
            if "file_path" in inp:
                return f"{name}: {Path(str(inp['file_path'])).name}"
            return name
    return ""


def detect_activity(
    root: Path | None = None, now: float | None = None
) -> ActivitySignal:
    """Classify the user's current coding activity. Pure read — no side effects."""
    root = root or (Path.home() / ".claude" / "projects")
    now = now if now is not None else time.time()

    sess = _latest_session(root)
    if sess is None:
        return ActivitySignal("no_session", "", 0.0, "")

    try:
        idle = now - sess.stat().st_mtime
    except OSError:
        idle = 0.0

    if idle > IDLE_AFTER_SECS:
        return ActivitySignal("idle", "", idle, sess.stem)

    kind = "idle"
    detail = ""
    for line in reversed(_tail_lines(sess)):
        try:
            d = json.loads(line)
        except Exception:
            continue
        t = d.get("type")
        if t not in ("assistant", "user"):
            continue
        content = (d.get("message") or {}).get("content")
        kinds = (
            [c.get("type") for c in content if isinstance(c, dict)]
            if isinstance(content, list)
            else []
        )
        if t == "assistant" and "tool_use" in kinds:
            kind, detail = "tool_running", _tool_detail(content)
            break
        if t == "user" and "tool_result" in kinds:
            kind = "thinking"
            break
        if t == "assistant" and "text" in kinds:
            kind = "awaiting_user"
            break
        if t == "user":
            kind = "thinking"
            break

    return ActivitySignal(kind, detail, idle, sess.stem)
