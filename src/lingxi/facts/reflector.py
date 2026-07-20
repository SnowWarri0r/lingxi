"""Tree-of-thought reflection, first-person AS Aria.

When triggered: pull recent N facts, ask Aria (LLM) to generate 3-5
high-level questions worth thinking about, then for each question pull
relevant facts and have Aria answer it with one compact insight. Each
answer is written as a `pattern` fact with high importance.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta

from lingxi.facts.models import Fact, FactType, Source
from lingxi.facts.retriever import FactQuery, FactRetriever
from lingxi.facts.writers.inference import InferenceWriter
from lingxi.providers.base import LLMProvider


# {self} = persona self-context. Depth follows the persona — a writer reflects
# deeply, a catgirl notices a light cat-level thing. Don't force introspection.
_SYSTEM_TMPL = "{self} 你在安静地回看自己最近的日子。"


_QUESTIONS_PROMPT = """我看自己最近经历的这些事，有没有什么**值得停下来想一下**的？
想到啥算啥，深浅看你是什么样的人——可能是反复出现的事、心情、跟谁的关系、
对自己的一个发现，也可能就是件一直惦记着的小事。小事就好，越具体越实在。

比如（看你自己的口吻和深浅）：
  - "最近老想着 X"
  - "我跟 X 好像有点不一样了"
  - "有件事一直搁我心里"

写 3-5 个。输出 JSON array of strings，每条就是一个问题，用你自己平时会想的措辞。

我最近经历的事：
{facts_block}
"""

_ANSWER_PROMPT = """问题：{q}

我手头有这些跟问题相关的事：
{facts_block}

我现在想一下这个问题，写**一条洞见**——浓缩，能补上事实之间的关系或趋势。
写你由这件事新看到的关于自己的一点。
1-2 句，用我自己想事情时的口语语气。
"""


class Reflector:
    def __init__(
        self,
        llm: LLMProvider,
        retriever: FactRetriever,
        inference_writer: InferenceWriter,
        model: str | None = None,
        min_facts: int = 10,
        recent_window: int = 100,
        per_question_limit: int = 15,
        no_pattern_lookback_hours: float = 24.0,
        persona=None,
    ):
        self._llm = llm
        self._retriever = retriever
        self._writer = inference_writer
        self._model = model
        from lingxi.persona.self_context import build_self_context
        self._self_ctx = (build_self_context(persona)
                          if persona is not None else "你是 Aria。")
        self._min_facts = min_facts
        self._recent_window = recent_window
        self._per_q_limit = per_question_limit
        self._no_pattern_lookback = timedelta(hours=no_pattern_lookback_hours)

    async def reflect(self) -> None:
        # Reflect on EVENTS only. Feeding past PATTERN facts back in makes the
        # reflector ruminate on its own reflections — it spirals into abstract
        # navel-gazing fixated on one theme (observed: days of "数心跳/在场"
        # phenomenology). Concrete daily events keep reflection grounded.
        # Watermark: only events newer than the latest pattern — each life
        # moment gets reflected on at most once. Without this the recent-N
        # window re-served the same old events on every run, regrowing the
        # same theme even after pattern inputs were excluded.
        #
        # When no pattern exists (fresh persona, or after a manual pattern
        # purge), there's no watermark to bound the window — reading the full
        # recent-N would re-chew whatever history is present. Fall back to a
        # recent time floor so a purge doesn't reopen the door to old events.
        latest_pattern = await self._retriever._store.query(
            subject="aria", type=FactType.PATTERN, limit=1
        )
        watermark = (latest_pattern[0].ts if latest_pattern
                     else datetime.now() - self._no_pattern_lookback)
        recent = await self._retriever._store.query(
            subject="aria", type=FactType.EVENT, since=watermark,
            limit=self._recent_window,
        )
        if len(recent) < self._min_facts:
            return

        questions = await self._generate_questions(recent)
        if not questions:
            return

        for q in questions:
            relevant = await self._retriever.fetch(
                FactQuery(subject="aria", type=FactType.EVENT,
                          semantic=q, since=watermark,
                          limit=self._per_q_limit)
            )
            insight = await self._answer(q, relevant)
            if not insight:
                continue
            now = datetime.now()
            pattern = Fact(
                subject="aria",
                content=insight,
                source=Source.LLM_INFERRED,
                type=FactType.PATTERN,
                ts=now,
                importance=8,
                # Patterns age out so they can't accumulate into a self-
                # reinforcing pile that dominates planning/proactive forever.
                expires_at=now + timedelta(days=14),
                tags=[f"reflection_question:{q[:80]}"],
            )
            await self._writer.write_skip_scorer(pattern, trigger_observation=False)

    async def _generate_questions(self, recent: list[Fact]) -> list[str]:
        facts_block = "\n".join(f"  - {f.content}" for f in recent[-50:])
        prompt = _QUESTIONS_PROMPT.format(facts_block=facts_block)
        try:
            kwargs = {"model": self._model} if self._model else {}
            response = await self._llm.complete(
                messages=[{"role": "user", "content": prompt}],
                system=_SYSTEM_TMPL.format(self=self._self_ctx),
                max_tokens=400,
                temperature=0.7,
                _debug_purpose="reflection_questions",
                **kwargs,
            )
            data = json.loads(_strip_fences(response.content))
            if isinstance(data, list):
                return [str(q).strip() for q in data if str(q).strip()][:5]
        except Exception as e:
            print(f"[reflector] question gen failed: {e}", flush=True)
        return []

    async def _answer(self, q: str, facts: list[Fact]) -> str:
        if not facts:
            return ""
        facts_block = "\n".join(f"  - {f.content}" for f in facts)
        prompt = _ANSWER_PROMPT.format(q=q, facts_block=facts_block)
        try:
            kwargs = {"model": self._model} if self._model else {}
            response = await self._llm.complete(
                messages=[{"role": "user", "content": prompt}],
                system=_SYSTEM_TMPL.format(self=self._self_ctx),
                max_tokens=200,
                temperature=0.7,
                _debug_purpose="reflection_answer",
                **kwargs,
            )
            return response.content.strip()
        except Exception as e:
            print(f"[reflector] answer failed for q={q!r}: {e}", flush=True)
            return ""


def _strip_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()
