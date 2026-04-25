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


async def test_seeds_always_present_when_retriever_populated(tmp_path: Path):
    """Spec §6.3: seeds must appear as baseline even when retriever has many samples.

    Populates the store with 10 high-similarity user_correction samples so that
    a naive k=6 retrieve-only approach would displace seeds.  The fixed engine
    must always inject seeds first, then retrieved samples.
    """
    store = FewShotStore(data_dir=tmp_path, embedding_dim=16)
    await store.init()

    # Add 10 user_correction samples all very similar to each other
    for i in range(10):
        s = FewShotSample(
            id=f"user-correction-{i}",
            inner_thought=f"USER-INNER-{i}",
            corrected_speech=f"USER-SPEECH-{i}",
            context_summary=f"USER-SCENE-{i}",
            source="user_correction",
        )
        await store.add(s, embedding=_embed(f"USER-INNER-{i}"))

    llm = FakeLLM()
    retriever = FewShotRetriever(store=store, embedder=llm)
    engine = ConversationEngine(
        persona=PersonaConfig(name="T", identity=Identity(full_name="T")),
        llm_provider=llm,
        memory_manager=MemoryManager(data_dir=str(tmp_path / "mem"), long_term_backend="json"),
        fewshot_store=store,
        fewshot_retriever=retriever,
    )

    await engine.chat("USER-INNER-0", channel="cli", recipient_id="tester")

    flat = str(llm.last_messages)

    # Baseline seeds from _phase0_seed_fewshots must always appear
    # (these are the hardcoded corrected_speech strings from Phase 0 seeds)
    assert "哦？啥机械？" in flat, "Phase-0 seed must be present as baseline anchor"
