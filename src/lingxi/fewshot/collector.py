"""Annotation collector — converts user feedback into FewShotSamples."""

from __future__ import annotations

import uuid
from typing import Protocol

from lingxi.fewshot.models import AnnotationTurn, FewShotSample
from lingxi.fewshot.store import AnnotationStore, FewShotStore


class Embedder(Protocol):
    async def embed(self, text: str) -> list[float]: ...


class Summarizer(Protocol):
    async def summarize(self, turn: AnnotationTurn) -> tuple[str, list[str]]: ...


class AnnotationCollector:
    """Front-end for three kinds of feedback.

    - record_positive: confirmed-good turn → pool as source=positive
    - record_negative: flag only, no pool write (wait for correction)
    - record_correction: negative + target speech → pool as source=user_correction
    """

    def __init__(
        self,
        annotation_store: AnnotationStore,
        fewshot_store: FewShotStore,
        embedder: Embedder,
        summarizer: Summarizer,
    ):
        self.annotations = annotation_store
        self.pool = fewshot_store
        self.embedder = embedder
        self.summarizer = summarizer

    async def record_positive(self, turn_id: str) -> FewShotSample:
        turn = await self._get_or_raise(turn_id)
        await self.annotations.update_annotation(turn_id, kind="positive")

        summary, tags = await self.summarizer.summarize(turn)
        sample = FewShotSample(
            id=f"pos-{uuid.uuid4().hex[:12]}",
            inner_thought=turn.inner_thought,
            original_speech=None,
            corrected_speech=turn.speech,
            context_summary=summary,
            tags=tags,
            recipient_key=turn.recipient_key,
            source="positive",
        )
        embedding = await self.embedder.embed(sample.inner_thought or sample.context_summary)
        await self.pool.add(sample, embedding=embedding)
        return sample

    async def record_negative(self, turn_id: str) -> None:
        await self._get_or_raise(turn_id)
        await self.annotations.update_annotation(turn_id, kind="negative")

    async def record_correction(self, turn_id: str, correction: str) -> FewShotSample:
        turn = await self._get_or_raise(turn_id)
        await self.annotations.update_annotation(
            turn_id, kind="negative", correction=correction,
        )

        summary, tags = await self.summarizer.summarize(turn)
        sample = FewShotSample(
            id=f"cor-{uuid.uuid4().hex[:12]}",
            inner_thought=turn.inner_thought,
            original_speech=turn.speech,
            corrected_speech=correction,
            context_summary=summary,
            tags=tags,
            recipient_key=turn.recipient_key,
            source="user_correction",
        )
        embedding = await self.embedder.embed(sample.inner_thought or sample.context_summary)
        await self.pool.add(sample, embedding=embedding)
        return sample

    async def _get_or_raise(self, turn_id: str) -> AnnotationTurn:
        turn = await self.annotations.get_turn(turn_id)
        if turn is None:
            raise KeyError(turn_id)
        return turn
