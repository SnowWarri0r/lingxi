"""Parse JSON out of an LLM reply, tolerating unescaped inner quotes.

Every JSON-returning prompt in this codebase asks for a Chinese `reason` /
`insight` / `note` string, and Chinese writing marks emphasis with quotes.
The model reaches for ASCII `"` to do it and hands back

    {"reason": "还没到"真正突破"的量级"}

which is a hard JSON syntax error. Measured over 2 872 logged JSON calls:
33% of importance_scorer, 19% of daily_planner, 10% of reflection_questions
and 9% of orchestrator replies failed on exactly this — 525 of 528 scorer
failures were `Expecting ',' delimiter`, and 524 of them ended in a proper
closing bracket, so the payloads were complete and only locally malformed.
Each failure silently dropped a whole batch to defaults.

The repair walks the text and re-decides, for each `"` inside a string,
whether it terminates the string: it does only when the next non-space
character is one that may legally follow a string value (`,` `:` `}` `]`)
or the input ends. Anything else is emphasis, and gets escaped.

This is deliberately narrow. Truncated output stays broken — a cut-off
reply has lost content, and guessing at the remainder would invent data
rather than recover it. Callers keep their existing fallback.
"""

from __future__ import annotations

import json
import re
from typing import Any

_CLOSERS = {",", ":", "}", "]"}


def strip_fences(text: str) -> str:
    """Drop a leading ```json / trailing ``` wrapper."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def escape_inner_quotes(text: str) -> str:
    """Escape `"` that sit inside a string value instead of ending it."""
    out: list[str] = []
    in_string = False
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if not in_string:
            out.append(ch)
            if ch == '"':
                in_string = True
            i += 1
            continue
        if ch == "\\":
            # Copy the escape pair through untouched.
            out.append(ch)
            if i + 1 < n:
                out.append(text[i + 1])
            i += 2
            continue
        if ch == '"':
            j = i + 1
            while j < n and text[j] in " \t\r\n":
                j += 1
            if j >= n or text[j] in _CLOSERS:
                out.append(ch)
                in_string = False
            else:
                out.append('\\"')
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def loads(text: str) -> Any:
    """json.loads on an LLM reply, repairing inner quotes if needed.

    Raises json.JSONDecodeError when the payload is broken beyond stray
    quotes (truncation, missing brackets) so callers keep their fallback.
    """
    cleaned = strip_fences(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return json.loads(escape_inner_quotes(cleaned))
