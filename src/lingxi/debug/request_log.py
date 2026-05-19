"""Full-fidelity LLM request/response logger.

Writes one JSON record per LLM call to a daily-rotated JSONL file. The
record contains the EXACT strings that went over the wire — system
prompt, every message in the history, and the response text — so we can
diagnose "why did the model say X" by reading what it was actually given.

Opt-in via env var `LINGXI_DEBUG_LLM=1` (or `=on`/`true`). Off by default.

Storage: `${MEMORY_DATA_DIR}/debug/llm_requests/YYYY-MM-DD.jsonl`
(falls back to `./data/debug/llm_requests/` when env var unset).

Record schema (one JSON object per line):
{
  "ts": "2026-05-19T11:14:15.123",
  "purpose": "chat_full|think|compress|biography_select|unknown",
  "model": "claude-sonnet-4-...",
  "system": "...full system prompt...",
  "messages": [{"role": "...", "content": "..."}],
  "response": "...full output text...",
  "usage": {"input_tokens": N, "output_tokens": M},
  "duration_ms": 1234,
  "request_id": "uuid"
}

Inspection: `tools/inspect_llm.py` (tail / by purpose / by request_id).
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

# Single writer lock — JSONL append needs atomic line writes
_WRITE_LOCK = threading.Lock()


def is_enabled() -> bool:
    """Whether request logging is on."""
    v = os.environ.get("LINGXI_DEBUG_LLM", "").strip().lower()
    return v in ("1", "on", "true", "yes")


def _log_dir() -> Path:
    base = os.environ.get("MEMORY_DATA_DIR", "./data/memory")
    # MEMORY_DATA_DIR points at the memory subdir; debug lives alongside it
    parent = Path(base).expanduser().resolve().parent
    return parent / "debug" / "llm_requests"


def _today_path() -> Path:
    d = _log_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{datetime.now().strftime('%Y-%m-%d')}.jsonl"


def log_request(
    *,
    system: str | None,
    messages: list[dict[str, Any]],
    response_text: str,
    model: str,
    usage: dict[str, int] | None = None,
    duration_ms: int | None = None,
    purpose: str = "unknown",
    extra: dict[str, Any] | None = None,
) -> str | None:
    """Append one record. Returns request_id (or None when disabled).

    Errors are swallowed — debug logging must never break the chat path.
    """
    if not is_enabled():
        return None
    try:
        rid = uuid.uuid4().hex[:12]
        record = {
            "ts": datetime.now().isoformat(timespec="milliseconds"),
            "request_id": rid,
            "purpose": purpose,
            "model": model,
            "system": system or "",
            "messages": messages,
            "response": response_text,
            "usage": usage or {},
            "duration_ms": duration_ms,
        }
        if extra:
            record["extra"] = extra
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with _WRITE_LOCK:
            with open(_today_path(), "a", encoding="utf-8") as f:
                f.write(line)
        return rid
    except Exception as e:
        # Don't propagate — debug must never break prod path
        print(f"[debug.request_log] write failed (non-fatal): {e}")
        return None
