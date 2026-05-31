import pytest
from pathlib import Path

from lingxi.stickers.store import StickerStore
from lingxi.conversation.engine import ConversationEngine
from lingxi.memory.manager import MemoryManager
from lingxi.persona.models import PersonaConfig, Identity


@pytest.mark.asyncio
async def test_engine_accepts_sticker_store(tmp_path):
    sstore = StickerStore(Path(tmp_path) / "stickers.db")
    await sstore.init()

    class _LLM:
        async def complete(self, **kw): ...

    eng = ConversationEngine(
        persona=PersonaConfig(name="Aria", identity=Identity(full_name="Aria")),
        llm_provider=_LLM(),
        memory_manager=MemoryManager(data_dir=str(Path(tmp_path) / "mem")),
        sticker_store=sstore,
    )
    assert eng.sticker_store is sstore
    assert eng._pending_stickers == {}
