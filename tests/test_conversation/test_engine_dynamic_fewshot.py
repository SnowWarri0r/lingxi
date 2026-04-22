"""Tests that engine uses FewShotRetriever when available."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lingxi.conversation.engine import ConversationEngine
from lingxi.fewshot.models import FewShotSample
from lingxi.fewshot.retriever import FewShotRetriever
from lingxi.fewshot.store import FewShotStore
from lingxi.memory.manager import MemoryManager
from lingxi.persona.models import Identity, PersonaConfig
from lingxi.providers.base import CompletionResult, LLMProvider, StreamChunk


class FakeLLM(LLMProvider):
    def __init__(self):
        self.last_messages = None

    async def complete(self, messages, system=None, max_tokens=4096,
                       temperature=0.7, top_p=None, prefill="", **kwargs):
        self.last_messages = list(messages)
        return CompletionResult(content=f"{prefill}嗯")

    async def complete_stream(self, messages, system=None, max_tokens=4096,
                              temperature=0.7, top_p=None, prefill="", **kwargs):
        yield StreamChunk(content="嗯", is_final=True)

    async def embed(self, text):
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        v = rng.normal(size=16)
        v = v / np.linalg.norm(v)
        return v.tolist()


def _embed(text: str, dim: int = 16) -> list[float]:
    rng = np.random.default_rng(abs(hash(text)) % (2**32))
    v = rng.normal(size=dim)
    v = v / np.linalg.norm(v)
    return v.tolist()


async def test_engine_uses_retriever_when_available(tmp_path: Path):
    store = FewShotStore(data_dir=tmp_path, embedding_dim=16)
    await store.init()

    # Seed one unique sample; engine should surface it in the prior-turn injection
    sample = FewShotSample(
        id="unique-signal",
        inner_thought="UNIQUE-SIGNAL",
        corrected_speech="UNIQUE-SPEECH",
        context_summary="UNIQUE-SCENE",
        source="seed",
    )
    await store.add(sample, embedding=_embed("UNIQUE-SIGNAL"))

    llm = FakeLLM()
    retriever = FewShotRetriever(store=store, embedder=llm)
    engine = ConversationEngine(
        persona=PersonaConfig(name="T", identity=Identity(full_name="T")),
        llm_provider=llm,
        memory_manager=MemoryManager(data_dir=str(tmp_path / "mem"), long_term_backend="json"),
        fewshot_store=store,
        fewshot_retriever=retriever,
    )

    # Trigger chat with text that embeds close to "UNIQUE-SIGNAL"
    await engine.chat("UNIQUE-SIGNAL", channel="cli", recipient_id="tester")

    # The prior-turn few-shot block should contain our unique sample's
    # context_summary and corrected_speech somewhere in the messages.
    flat = str(llm.last_messages)
    assert "UNIQUE-SCENE" in flat
    assert "UNIQUE-SPEECH" in flat
