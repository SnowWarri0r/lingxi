"""LLM-assisted summarizer for AnnotationTurn → (context_summary, tags).

Falls back to heuristic extraction if the LLM isn't available.
"""

from __future__ import annotations

from lingxi.fewshot.models import AnnotationTurn
from lingxi.providers.base import LLMProvider


_SUMMARIZER_PROMPT = """你在帮助标注一段对话。下面是用户输入、Aria 的内心想法、和 Aria 实际说的话。

请输出一行"场景一句话总结"（≤20 字）和 2-4 个"场景标签"（每个标签 1-4 字）。

格式：
场景：<一句话>
标签：<tag1>,<tag2>,...

对话：
用户：{user_message}
Aria 想：{inner_thought}
Aria 说：{speech}
"""


class AnnotationSummarizer:
    def __init__(self, llm: LLMProvider):
        self.llm = llm

    async def summarize(self, turn: AnnotationTurn) -> tuple[str, list[str]]:
        prompt = _SUMMARIZER_PROMPT.format(
            user_message=turn.user_message,
            inner_thought=turn.inner_thought or "(无)",
            speech=turn.speech,
        )
        try:
            result = await self.llm.complete(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.3,
            )
            return _parse_summarizer_output(result.content, fallback=turn)
        except Exception:
            # Heuristic fallback
            return (
                turn.user_message[:20] or "(无场景)",
                [],
            )


def _parse_summarizer_output(text: str, fallback: AnnotationTurn) -> tuple[str, list[str]]:
    summary = ""
    tags: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("场景") and (":" in line or "：" in line):
            summary = line.split("：", 1)[-1].split(":", 1)[-1].strip()
        elif line.startswith("标签") and (":" in line or "：" in line):
            raw = line.split("：", 1)[-1].split(":", 1)[-1]
            tags = [t.strip() for t in raw.replace("，", ",").split(",") if t.strip()]
    if not summary:
        summary = fallback.user_message[:20] or "(无场景)"
    return summary, tags[:4]
