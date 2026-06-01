"""Turn (title, real replies) into FewShotSamples — deterministic, no LLM.

corrected_speech is the verbatim real reply; context_summary is the real
thread title; tags come from a keyword map. inner_thought is left empty so no
model-generated voice enters the demonstration pool.
"""

from __future__ import annotations

from lingxi.fewshot.models import FewShotSample

# Ordered: specific/emotional keys first, generic ("分享"/"今天") last —
# first match wins. Keep emotion above "分享" so a "快乐分享" thread tags 开心.
_TAG_MAP = [
    ("星空", "星空"), ("星星", "星空"), ("流星", "星空"),
    ("失眠", "失眠"), ("睡不着", "失眠"), ("熬夜", "深夜"), ("还没睡", "深夜"),
    ("倾诉", "倾诉"), ("内耗", "内耗"), ("emo", "emo"),
    ("孤独", "孤独"), ("孤单", "孤独"), ("一个人", "独处"), ("独处", "独处"),
    ("自由职业", "自由"),
    ("快乐", "开心"), ("开心", "开心"), ("高兴", "开心"),
    ("治愈", "温暖"), ("抱抱", "温暖"), ("拥抱", "温暖"), ("幸福", "温暖"),
    ("喜欢", "温暖"),
    ("焦虑", "焦虑"), ("难受", "难过"), ("哭", "难过"),
    ("好奇", "好奇"), ("想问", "好奇"),
    ("分享", "分享"), ("今天", "日常"), ("上班", "日常"),
]


def _tags(title: str, line: str) -> list[str]:
    hay = title + " " + line
    for kw, tag in _TAG_MAP:
        if kw in hay:
            return [tag]
    return ["日常"]


def build_samples(title: str, replies: list[str], *, topic_id: str
                  ) -> list[FewShotSample]:
    out: list[FewShotSample] = []
    for i, line in enumerate(replies):
        out.append(FewShotSample(
            id=f"corpus-douban-{topic_id}-{i}",
            inner_thought="",
            corrected_speech=line,
            context_summary=title,
            tags=_tags(title, line),
            recipient_key=None,
            source="corpus",
        ))
    return out
