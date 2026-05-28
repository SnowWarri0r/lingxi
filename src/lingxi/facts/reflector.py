"""Tree-of-thought reflection, first-person AS Aria.

When triggered: pull recent N facts, ask Aria (LLM) to generate 3-5
high-level questions worth thinking about, then for each question pull
relevant facts and have Aria answer it with one compact insight. Each
answer is written as a `pattern` fact with high importance.
"""

from __future__ import annotations

import json
import re
from datetime import datetime

from lingxi.facts.models import Fact, FactType, Source
from lingxi.facts.retriever import FactQuery, FactRetriever
from lingxi.facts.writers.inference import InferenceWriter
from lingxi.providers.base import LLMProvider


_SYSTEM = "你是 Aria，正在安静地回看自己最近的生活。"


_QUESTIONS_PROMPT = """我看自己最近经历的这些事，有没有什么**值得停下来想一想**的问题？
不要琐碎的（"我今天吃了什么"这种没意义），要那些能让我**真的反思**的——
关于我最近的模式、情绪走向、和别人的关系变化、对自己的认知。

比如：
  - "我最近在工作上是不是有点提不起劲了？"
  - "我跟 X 的相处方式好像有点变了，是哪里变了？"
  - "最近反复在我脑子里冒出来的事是什么？"

写 3-5 个。输出 JSON array of strings，每条就是一个问题，用我自己平时会想的措辞。

我最近经历的事：
{facts_block}
"""

_ANSWER_PROMPT = """问题：{q}

我手头有这些跟问题相关的事：
{facts_block}

我现在想一下这个问题，写**一条洞见**——浓缩，能补上事实之间的关系或趋势。
不要复述事实本身（"我最近忙工作"是废话）。
1-2 句，用我自己想事情时的语气，不要书面化。
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
    ):
        self._llm = llm
        self._retriever = retriever
        self._writer = inference_writer
        self._model = model
        self._min_facts = min_facts
        self._recent_window = recent_window
        self._per_q_limit = per_question_limit

    async def reflect(self) -> None:
        recent = await self._retriever._store.query(
            subject="aria", limit=self._recent_window
        )
        if len(recent) < self._min_facts:
            return

        questions = await self._generate_questions(recent)
        if not questions:
            return

        for q in questions:
            relevant = await self._retriever.fetch(
                FactQuery(subject="aria", semantic=q, limit=self._per_q_limit)
            )
            insight = await self._answer(q, relevant)
            if not insight:
                continue
            pattern = Fact(
                subject="aria",
                content=insight,
                source=Source.LLM_INFERRED,
                type=FactType.PATTERN,
                ts=datetime.now(),
                importance=8,
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
                system=_SYSTEM,
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
                system=_SYSTEM,
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
