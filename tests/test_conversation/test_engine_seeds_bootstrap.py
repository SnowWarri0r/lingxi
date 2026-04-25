"""Tests that seeds are loaded into FewShotStore on first engine init."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lingxi.conversation.engine import ConversationEngine
from lingxi.fewshot.store import FewShotStore
from lingxi.memory.manager import MemoryManager
from lingxi.persona.models import Identity, PersonaConfig
from lingxi.providers.base import CompletionResult, LLMProvider, StreamChunk


class FakeLLM(LLMProvider):
    async def complete(self, messages, system=None, max_tokens=4096,
                       temperature=0.7, top_p=None, prefill="", **kwargs):
        return CompletionResult(content=f"{prefill}ok")

    async def complete_stream(self, messages, system=None, max_tokens=4096,
                              temperature=0.7, top_p=None, prefill="", **kwargs):
        yield StreamChunk(content="ok", is_final=True)

    async def embed(self, text: str):
        # Deterministic 16-d embedding
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        v = rng.normal(size=16)
        v = v / np.linalg.norm(v)
        return v.tolist()


async def test_seeds_bootstrap_populates_store(tmp_path: Path):
    store = FewShotStore(data_dir=tmp_path, embedding_dim=16)
    await store.init()
    assert await store.count() == 0

    persona = PersonaConfig(name="T", identity=Identity(full_name="T"))
    engine = ConversationEngine(
        persona=persona,
        llm_provider=FakeLLM(),
        memory_manager=MemoryManager(data_dir=str(tmp_path / "mem"), long_term_backend="json"),
        fewshot_store=store,
    )
    # Bootstrap is called explicitly; running it is idempotent
    await engine.bootstrap_fewshot_seeds()
    assert await store.count() == 10

    # Running again should not duplicate
    await engine.bootstrap_fewshot_seeds()
    assert await store.count() == 10
