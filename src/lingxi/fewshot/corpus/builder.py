"""Turn (title, real replies) into FewShotSamples — deterministic, no LLM.

corrected_speech is the verbatim real reply; context_summary is the real
thread title; tags come from a keyword map. inner_thought is left empty so no
model-generated voice enters the demonstration pool.
"""

from __future__ import annotations

from lingxi.fewshot.models import FewShotSample

_TAG_MAP = [
    ("倾诉", "倾诉"), ("孤独", "孤独"), ("孤单", "孤独"), ("内耗", "内耗"),
    ("失眠", "失眠"), ("emo", "emo"), ("难受", "难过"), ("哭", "难过"),
    ("分享", "分享"), ("独处", "独处"), ("一个人", "独处"), ("焦虑", "焦虑"),
    ("幸福", "温暖"), ("喜欢", "温暖"),
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
