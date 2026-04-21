"""Structured turn output schema.

LLM output format:
    <speech text>
    ===META===
    {
      "expression": "...",
      "action": "...",
      "mood": "...",
      "emotion": {"dim": 0.7, ...},
      "memory_writes": ["..."],
      "plan_updates": ["..."],
      "inner": "..."
    }

Speech comes first (streams naturally). Metadata JSON follows a clear
delimiter and is parsed as a whole. No regex multi-tag parsing.
"""

from __future__ import annotations

import json

from pydantic import BaseModel, Field

from lingxi.conversation.response_cleaner import clean_speech


META_DELIMITER = "===META==="


class TurnOutput(BaseModel):
    """All parallel outputs from a single conversation turn."""

    turn_id: str = ""

    # Spoken content (what text channels render)
    speech: str = ""

    # Expression / facial / vocal tone (for avatar, TTS)
    expression: str = ""

    # Physical action / gesture
    action: str = ""

    # Mood / emotion
    mood_label: str = ""
    emotion_deltas: dict[str, float] = Field(default_factory=dict)

    # Memory & planning
    memory_writes: list[str] = Field(default_factory=list)
    plan_updates: list[str] = Field(default_factory=list)

    # Inner thought (not spoken)
    inner_thought: str = ""

    # Raw LLM output (for debugging)
    raw: str = ""


def parse_turn_output(raw: str) -> TurnOutput:
    """Split raw output on META_DELIMITER; parse trailing JSON.

    Graceful degradation: if no delimiter or JSON invalid, treat everything
    as speech (so we never lose the main content).
    """
    out = TurnOutput(raw=raw)

    if META_DELIMITER not in raw:
        out.speech = clean_speech(raw.strip())
        return out

    parts = raw.split(META_DELIMITER, 1)
    speech_part = parts[0].strip()
    meta_part = parts[1].strip() if len(parts) > 1 else ""

    # Clean narration/meta that leaked into speech despite the JSON format
    out.speech = clean_speech(speech_part)

    if not meta_part:
        return out

    # Find the first `{` ... last `}` to tolerate stray chars
    first = meta_part.find("{")
    last = meta_part.rfind("}")
    if first == -1 or last == -1 or last <= first:
        return out

    try:
        data = json.loads(meta_part[first : last + 1])
    except (json.JSONDecodeError, ValueError):
        return out

    if not isinstance(data, dict):
        return out

    out.expression = str(data.get("expression", "") or "")[:200]
    out.action = str(data.get("action", "") or "")[:200]
    out.mood_label = str(data.get("mood", "") or "")[:50]

    emotion = data.get("emotion")
    if isinstance(emotion, dict):
        for k, v in emotion.items():
            try:
                out.emotion_deltas[str(k)] = float(v)
            except (TypeError, ValueError):
                continue

    mw = data.get("memory_writes") or data.get("memory")
    if isinstance(mw, list):
        out.memory_writes = [str(x).strip() for x in mw if str(x).strip()]
    elif isinstance(mw, str) and mw.strip():
        out.memory_writes = [mw.strip()]

    pu = data.get("plan_updates") or data.get("plans")
    if isinstance(pu, list):
        out.plan_updates = [str(x).strip() for x in pu if str(x).strip()]
    elif isinstance(pu, str) and pu.strip():
        out.plan_updates = [pu.strip()]

    inner = data.get("inner") or data.get("inner_thought")
    if isinstance(inner, str):
        out.inner_thought = inner.strip()

    return out
