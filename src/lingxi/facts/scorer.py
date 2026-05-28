"""Batched, first-person importance scoring.

Aria (or the NPC, depending on subject) scores her own recent facts
by subjective "this matters to me" — the Generative Agents paper's
"poignancy" rating. NOT an external rater.

Batching: writers call score_one(fact) which returns a Future. The
scorer flushes the buffer when it hits batch_size or flush_seconds,
makes one LLM call scoring all queued facts.

Each persona (aria, each npc) gets its own bucket so one LLM call
doesn't mix voices.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass

from lingxi.facts.models import Fact, Source
from lingxi.providers.base import LLMProvider


DEFAULT_IMPORTANCE: dict[Source, int] = {
    Source.USER_STATED: 7,
    Source.BIOGRAPHY: 8,
    Source.LIFE_SIMULATED: 3,
    Source.NPC_TICKER: 4,
    Source.LLM_INFERRED: 5,
    Source.WORLD_FETCH: 3,
}


_ARIA_SYSTEM = "你是 Aria，正在回看自己最近经历的事。"
_NPC_SYSTEM_TEMPLATE = "你是 {name}，正在回看自己最近经历的事。"
_NEUTRAL_SYSTEM = "你正在回看最近一批事件并打分。"


_PROMPT_TEMPLATE = """我在给自己最近经历的事打分——这些事对**我**来说有多重要（1-10）。
1 = 完全琐碎（"喝了口水"），10 = 改变我和别人关系、或者改变我人生方向的事。
主观判断，不是客观新闻价值：
  - 我每天重复的作息 → 1-3
  - 普通工作进展 → 3-5
  - 跟我在意的人有情感交流 / 我自己情绪起伏 → 6-8
  - 关键关系变化 / 重大决定 / 真正触动到我的事 → 8-10

输入 {n} 条事件，输出 JSON array：
[{{"id": "...", "score": 1-10, "reason": "一句话——为什么对我来说是这个分"}}, ...]

事件：
{facts_block}
"""


@dataclass
class _PendingFact:
    fact: Fact
    future: asyncio.Future


def _resolve_system(subject: str) -> str:
    if subject == "aria":
        return _ARIA_SYSTEM
    if subject.startswith("npc:"):
        return _NPC_SYSTEM_TEMPLATE.format(name=subject.removeprefix("npc:"))
    return _NEUTRAL_SYSTEM


def _bucket_key(subject: str) -> str:
    """Group facts by persona so one LLM call doesn't mix voices."""
    if subject == "aria":
        return "aria"
    if subject.startswith("npc:"):
        return subject
    return "other"


class ImportanceScorer:
    def __init__(
        self,
        llm: LLMProvider,
        batch_size: int = 5,
        flush_seconds: float = 30.0,
        model: str | None = None,
    ):
        self._llm = llm
        self._batch_size = batch_size
        self._flush_seconds = flush_seconds
        self._model = model
        self._buckets: dict[str, list[_PendingFact]] = {}
        self._flush_tasks: dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()

    async def score_one(self, fact: Fact) -> int:
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        bucket = _bucket_key(fact.subject)
        async with self._lock:
            self._buckets.setdefault(bucket, []).append(_PendingFact(fact, future))
            if len(self._buckets[bucket]) >= self._batch_size:
                self._cancel_flush_timer(bucket)
                await self._flush(bucket)
            else:
                self._schedule_flush_timer(bucket)
        try:
            return await future
        except Exception:
            return DEFAULT_IMPORTANCE.get(fact.source, 5)

    def _schedule_flush_timer(self, bucket: str) -> None:
        if bucket in self._flush_tasks and not self._flush_tasks[bucket].done():
            return
        self._flush_tasks[bucket] = asyncio.create_task(self._flush_after_delay(bucket))

    def _cancel_flush_timer(self, bucket: str) -> None:
        task = self._flush_tasks.pop(bucket, None)
        if task and not task.done():
            task.cancel()

    async def _flush_after_delay(self, bucket: str) -> None:
        try:
            await asyncio.sleep(self._flush_seconds)
        except asyncio.CancelledError:
            return
        async with self._lock:
            if self._buckets.get(bucket):
                await self._flush(bucket)

    async def _flush(self, bucket: str) -> None:
        """Caller must hold self._lock."""
        pending = self._buckets.pop(bucket, [])
        if not pending:
            return
        system = _resolve_system(pending[0].fact.subject)
        facts_block = "\n".join(
            f"[{i+1}] id={p.fact.id} type={p.fact.type.value} content=\"{p.fact.content}\""
            for i, p in enumerate(pending)
        )
        prompt = _PROMPT_TEMPLATE.format(n=len(pending), facts_block=facts_block)
        try:
            kwargs = {"model": self._model} if self._model else {}
            response = await self._llm.complete(
                messages=[{"role": "user", "content": prompt}],
                system=system,
                max_tokens=400,
                temperature=0.3,
                _debug_purpose="importance_scorer",
                **kwargs,
            )
            data = json.loads(_strip_json_fences(response.content))
            scores_by_id = {
                item["id"]: int(item["score"])
                for item in data
                if isinstance(item, dict) and "id" in item and "score" in item
            }
            for p in pending:
                score = scores_by_id.get(p.fact.id)
                if score is None or not (1 <= score <= 10):
                    score = DEFAULT_IMPORTANCE.get(p.fact.source, 5)
                if not p.future.done():
                    p.future.set_result(score)
        except Exception as e:
            print(f"[scorer] LLM batch failed, using defaults: {e}", flush=True)
            for p in pending:
                if not p.future.done():
                    p.future.set_result(DEFAULT_IMPORTANCE.get(p.fact.source, 5))


def _strip_json_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()
