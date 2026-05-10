"""LLM-based biography event selector (CC-style).

Replaces BiographyRetriever's embedding-similarity retrieval with an
LLM-judgment layer. Pattern lifted from Claude Code's
memdir/findRelevantMemories: build a manifest of all candidate
biography events, give it to a fast model with the user's query +
tone hints, ask it to pick 0-2 events that are GENUINELY relevant.

Why this beats cosine similarity for our case:
- Embedding similarity at threshold 0.25 is loose; "苦中作乐" matches
  "准备下班" because both have temporal/state words. The retrieved
  fragment then leaks into a turn it shouldn't.
- The LLM understands NUANCE: "user is venting about work, this
  18-yo grandma death event is not the moment". Embedding cannot
  reason about social appropriateness or query lightness.
- No more keyword-marker maintenance: the heavy_content_marker filter
  in engine.py was a band-aid. The LLM filters by meaning instead.

Cost: 1 fast-model call per turn (~300ms with Haiku). Acceptable
given the alternative is leaking heavy biography into wrong turns.
"""

from __future__ import annotations

import json
import re
from typing import Any

from lingxi.persona.models import LifeEvent
from lingxi.providers.base import LLMProvider


_SELECTOR_SYSTEM_PROMPT = """你在帮 Aria 选她过去经历过的事，决定哪些值得在这一轮被想起。

**这些是 Aria 的人生记忆候选**——你只能从下面列表里选，不能编。

每条格式：[id] N岁·主题摘要

**任务**：根据用户当下的话 + 当下情绪/谈话状态，挑出 **0-2 条** 真的会自然想起的事。

**严格规则**：
1. **大多数情况返回空 list**——大部分对话不需要回忆过去
2. **不要凑数**——不确定就不选
3. **沉重事件**（失去/死亡/抑郁/想过结束）只有在用户**自己在谈这类话题时**才能选；否则**绝对不要**选
4. **被质问/被指责时**（用户说"好敷衍/什么关系/你都不"）→ 返回空 list（她在防御，不是分享）
5. **轻松对话**（聊天/分享日常）→ 只选轻松的事件，避开沉重的
6. **真的相关**意味着：这条事件能帮助 Aria 更真实地回应当下话题，而不是为了显得"丰富"硬塞

**输出格式**（严格 JSON，不要 markdown 包裹）：
```
{"selected": ["id1", "id2"]}
```
或者：
```
{"selected": []}
```"""


class BiographySelector:
    """Pick biography events relevant to the current turn via LLM judgment."""

    def __init__(
        self,
        events: list[LifeEvent],
        llm: LLMProvider,
        *,
        model: str | None = None,
    ):
        self.events = events
        self.llm = llm
        # Use the configured model (caller passes Haiku for speed).
        # None means fall back to the LLMProvider's default.
        self.model = model

    def append(self, event: LifeEvent) -> None:
        """Append a newly-acquired LifeEvent. Cheap — no embedding pass."""
        self.events.append(event)

    def _build_manifest(self) -> str:
        """Render one line per event: [id] N岁·content snippet."""
        lines: list[str] = []
        for idx, e in enumerate(self.events):
            content = (e.content or "").replace("\n", " ").strip()
            if len(content) > 60:
                content = content[:60] + "…"
            tags = ("、".join(e.tags[:3])) if e.tags else ""
            tag_part = f" #{tags}" if tags else ""
            lines.append(f"[{idx}] {e.age}岁·{content}{tag_part}")
        return "\n".join(lines)

    def _build_query_block(
        self,
        query: str,
        *,
        is_heavy: bool,
        is_confrontation: bool,
        recent_emotion: str | None,
    ) -> str:
        parts = [f"用户当下说的：{query.strip()}"]
        tone_bits: list[str] = []
        if is_heavy:
            tone_bits.append("用户在谈沉重话题")
        if is_confrontation:
            tone_bits.append("用户在质问/指责 Aria（防御场景）")
        if recent_emotion:
            tone_bits.append(f"Aria 当前情绪：{recent_emotion}")
        if tone_bits:
            parts.append("当下场景：" + "；".join(tone_bits))
        return "\n".join(parts)

    @staticmethod
    def _strip_json_fences(text: str) -> str:
        text = text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        return text.strip()

    async def select(
        self,
        query: str,
        *,
        is_heavy: bool = False,
        is_confrontation: bool = False,
        recent_emotion: str | None = None,
        max_events: int = 2,
    ) -> list[LifeEvent]:
        """Return up to `max_events` LifeEvents the LLM judges relevant.

        Returns empty list when:
        - no events at all
        - confrontation flag set (short-circuit, save the call)
        - LLM returns empty / parse fails
        """
        if not self.events or not query.strip():
            return []
        # Confrontation: don't even call the LLM. She's defending herself,
        # not opening up. Save the latency.
        if is_confrontation:
            return []

        manifest = self._build_manifest()
        user_block = self._build_query_block(
            query,
            is_heavy=is_heavy,
            is_confrontation=is_confrontation,
            recent_emotion=recent_emotion,
        )
        prompt = (
            f"{user_block}\n\n"
            f"候选 biography events（id 是中括号内数字）：\n{manifest}\n\n"
            f"输出 JSON，最多挑 {max_events} 条："
        )

        try:
            kwargs: dict = {}
            if self.model:
                kwargs["model"] = self.model
            result = await self.llm.complete(
                messages=[{"role": "user", "content": prompt}],
                system=_SELECTOR_SYSTEM_PROMPT,
                max_tokens=200,
                temperature=0.3,
                **kwargs,
            )
            text = self._strip_json_fences(
                result.content if hasattr(result, "content") else str(result)
            )
            data: Any = json.loads(text)
        except Exception as e:
            print(f"[bio_selector] LLM/parse failed: {e}", flush=True)
            return []

        if not isinstance(data, dict):
            return []
        selected_raw = data.get("selected", [])
        if not isinstance(selected_raw, list):
            return []

        result_events: list[LifeEvent] = []
        for raw_id in selected_raw[:max_events]:
            try:
                idx = int(str(raw_id).strip())
            except (ValueError, TypeError):
                continue
            if 0 <= idx < len(self.events):
                result_events.append(self.events[idx])

        return result_events
