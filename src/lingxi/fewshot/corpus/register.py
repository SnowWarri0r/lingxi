"""Register filter: keep only short, spoken-texture, first-person-ish lines.

Heuristic + deterministic. The goal is anti-翻译腔 cadence (碎句/省主语/语气词),
so we keep lines with spoken markers and drop essays, ads, links, agreement
filler. Returns the cleaned line or None to drop.
"""

from __future__ import annotations

import re

_MIN, _MAX = 4, 40
_SPOKEN = re.compile(r"[啊吧呢啦诶嗯喔哈呀嘛哦唉]|…|。。|\.\.\.|^[有就其实]")
_AGREEMENT = {"同意", "对", "对对", "是的", "+1", "赞", "同", "顶", "嗯嗯", "哈哈哈"}
_DROP_MARKERS = re.compile(
    r"https?://|[@#]|回复\s|￥|\d+元|链接|入手|测评|种草|推荐|优惠|代购|私信|vx|微信")
_EMOJI = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF]")


def clean_and_keep(line: str) -> str | None:
    line = (line or "").strip()
    if not line:
        return None
    if _DROP_MARKERS.search(line):
        return None
    bare = _EMOJI.sub("", line).strip()
    if not bare or bare in _AGREEMENT:
        return None
    if not (_MIN <= len(bare) <= _MAX):
        return None
    if not (_SPOKEN.search(bare) or (bare.endswith("。") and len(bare) <= 20)):
        return None
    return bare
