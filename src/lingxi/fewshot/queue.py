"""Rank recorded turns so the worst get annotated first.

The voice-anchor pool holds 43 generic seeds and 2 approved lines, while 198
recorded turns sit unannotated. Anchors are what keep her cadence attached to
real speech, and nothing tunable substitutes for having some — so the
bottleneck is attention, and attention should go to the turns that read least
like a person.

Ranking is a judgment call, so a model makes it, in batches. The heuristics
measured elsewhere in this codebase (comfort phrases, tilde, exclamation
density) are computed too, but only as a visible feature beside each turn:
they were derived from one persona on one week of logs and are far too crude
to order a queue by.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

_RANK_PROMPT = (
    "下面是一个中文角色在 IM 里发出的消息，每条带编号。\n"
    "给每条打一个 0-10 分：**读起来有多像 AI 助手在说话**，而不是一个真人在发微信。\n"
    "10 = 明显是 AI（客套、万能安慰、书面腔、过度热情、什么都要总结升华、"
    "每句都在关心对方）。\n"
    "0 = 完全像真人随手发的（可以是短句、残句、口语、没头没尾）。\n"
    "只输出 JSON 数组，每项 {\"i\": 编号, \"s\": 分数, \"why\": \"≤12字\"}。\n\n"
)

# Surface markers whose rates were measured across this persona's own IM
# replies versus its context-free desktop lines: comfort phrases 0% vs 22.5%,
# tilde 0% vs 8%. Shown, never used to sort.
_MARKERS = (
    ("客套", re.compile(r"辛苦了|加油|注意休息|早点休息|喝口水|歇会|多喝热水")),
    ("波浪", re.compile(r"[~～]")),
    ("三感叹", re.compile(r"！.*！.*！")),
    ("升华", re.compile(r"其实|这就是|人生|意义|说明你|真的很棒")),
)


@dataclass
class RankedTurn:
    turn_id: str
    speech: str
    user_message: str
    score: float
    why: str
    markers: list[str]


def surface_markers(text: str) -> list[str]:
    return [name for name, pat in _MARKERS if pat.search(text or "")]


def _parse_scores(raw: str) -> dict[int, tuple[float, str]]:
    """Read the model's array, defensively — a bad batch scores nothing."""
    cleaned = re.sub(r"```(?:json)?", "", raw or "")
    match = re.search(r"\[.*\]", cleaned, re.S)
    if not match:
        return {}
    try:
        rows = json.loads(match.group(0))
    except (json.JSONDecodeError, ValueError):
        return {}
    out: dict[int, tuple[float, str]] = {}
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict) or "i" not in row:
            continue
        try:
            out[int(row["i"])] = (float(row.get("s", 0)), str(row.get("why", "")))
        except (TypeError, ValueError):
            continue
    return out


async def rank_turns(turns: list, provider, *, batch: int = 20) -> list[RankedTurn]:
    """Score every turn, worst first. Turns the model skips are dropped.

    Dropping is deliberate: an unscored turn given a default would sort into
    the middle of the queue and quietly consume the attention this exists to
    direct.
    """
    ranked: list[RankedTurn] = []
    for start in range(0, len(turns), batch):
        chunk = turns[start:start + batch]
        listing = "\n".join(
            f"{i}. {t.speech}" for i, t in enumerate(chunk) if (t.speech or "").strip()
        )
        if not listing:
            continue
        try:
            result = await provider.complete(
                messages=[{"role": "user", "content": _RANK_PROMPT + listing}],
                max_tokens=2000, temperature=0.0,
                _debug_purpose="annotation_queue_rank",
            )
            scores = _parse_scores(result.content)
        except Exception as e:
            print(f"[queue] batch at {start} failed: {e}", flush=True)
            continue
        for i, turn in enumerate(chunk):
            if i not in scores:
                continue
            score, why = scores[i]
            ranked.append(RankedTurn(
                turn_id=turn.turn_id, speech=turn.speech,
                user_message=turn.user_message, score=score, why=why,
                markers=surface_markers(turn.speech),
            ))
    ranked.sort(key=lambda r: r.score, reverse=True)
    return ranked
