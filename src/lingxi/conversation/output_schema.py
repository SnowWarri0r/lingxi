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
import re

from pydantic import BaseModel, Field

from lingxi.conversation.response_cleaner import clean_speech


META_DELIMITER = "===META==="

# A single trailing `#表情 <情绪词>` line, which is how the single-pass
# responder actually asks for a sticker.
#
# The ===META=== JSON block below is the Claude-with-tools format and it
# still works there (74% of logged tool-loop turns carried one). The
# domestic single-pass responder emits it 0% of the time — measured across
# DeepSeek and doubao, and unchanged by strengthening the instruction. A
# JSON trailer fights everything else the prompt asks of it, which is to
# type short IM messages and stop. One short line does not, and lands at
# ~25% on an emotional turn and 0% on a flat one.
_STICKER_TAG = re.compile(r"^[ \t]*#\s*表情[ \t:：]+([^\s#]{1,20})[ \t]*$", re.M)


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

    # Sticker intent: a short mood/emotion description the persona wants to
    # send as a 表情包 this turn. The engine searches the sticker store with
    # this and resolves it to an actual image (empty = no sticker).
    sticker: str = ""

    # Raw LLM output (for debugging)
    raw: str = ""


def parse_turn_output(raw: str) -> TurnOutput:
    """Split raw output on META_DELIMITER; parse trailing JSON.

    Graceful degradation: if no delimiter or JSON invalid, treat everything
    as speech (so we never lose the main content).
    """
    out = TurnOutput(raw=raw)

    # Lift the sticker tag out first, before any early return — the common
    # case now is a reply with a tag and no ===META=== block at all. Last one
    # wins if several are written. Stripping it here is what keeps the marker
    # out of the chat window.
    tags = _STICKER_TAG.findall(raw)
    if tags:
        out.sticker = tags[-1].strip()[:60]
        raw_body = _STICKER_TAG.sub("", raw)
    else:
        raw_body = raw

    if META_DELIMITER not in raw_body:
        out.speech = clean_speech(raw_body.strip())
        return out

    parts = raw_body.split(META_DELIMITER, 1)
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

    # A JSON sticker field still wins if one is present (Claude path); the
    # tag is what the single-pass responder can actually produce.
    from_json = str(data.get("sticker", "") or "").strip()[:60]
    if from_json:
        out.sticker = from_json

    return out
