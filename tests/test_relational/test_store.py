"""Tests for RelationalMemoryStore — atomic per-recipient persistence."""

import pytest

from lingxi.relational.models import InsideJoke, RelationalMemory
from lingxi.relational.store import RelationalMemoryStore


@pytest.fixture
def store(tmp_path):
    return RelationalMemoryStore(tmp_path)


@pytest.mark.asyncio
async def test_load_missing_returns_blank(store):
    m = await store.load("feishu:none")
    assert m.recipient_key == "feishu:none"
    assert m.is_empty()


@pytest.mark.asyncio
async def test_save_and_load_roundtrip(store):
    initial = RelationalMemory(
        recipient_key="feishu:abc",
        pet_names=["笨蛋"],
        inside_jokes=[InsideJoke(phrase="蜘蛛会做梦的", origin="某次梦")],
    )
    await store.save(initial)
    loaded = await store.load("feishu:abc")
    assert loaded.pet_names == ["笨蛋"]
    assert loaded.inside_jokes[0].phrase == "蜘蛛会做梦的"


@pytest.mark.asyncio
async def test_unsafe_chars_in_recipient_key_handled(store):
    # Realistic recipient keys have ":" and possibly other special chars.
    # Store should sanitize for filename without losing the data.
    key = "feishu:oc_abc/def"
    m = RelationalMemory(recipient_key=key, pet_names=["x"])
    await store.save(m)
    loaded = await store.load(key)
    assert loaded.recipient_key == key
    assert loaded.pet_names == ["x"]


@pytest.mark.asyncio
async def test_update_memory_atomic_merge(store):
    # Pre-existing data
    initial = RelationalMemory(
        recipient_key="x",
        pet_names=["笨蛋"],
    )
    await store.save(initial)

    def _mutate(m):
        m.pet_names.append("老李")
        m.inside_jokes.append(InsideJoke(phrase="X", origin="Y"))

    await store.update_memory("x", _mutate)
    loaded = await store.load("x")
    assert set(loaded.pet_names) == {"老李", "笨蛋"}
    assert len(loaded.inside_jokes) == 1


@pytest.mark.asyncio
async def test_update_memory_serializes_concurrent_callers(store):
    # 10 concurrent updates each appending a name; final must have all 10
    import asyncio

    initial = RelationalMemory(recipient_key="x")
    await store.save(initial)

    async def add(i):
        def _mutate(m):
            m.pet_names.append(f"name{i}")
        await store.update_memory("x", _mutate)

    await asyncio.gather(*[add(i) for i in range(10)])
    loaded = await store.load("x")
    assert len(loaded.pet_names) == 10


@pytest.mark.asyncio
async def test_corrupted_file_falls_through_to_blank(store, tmp_path):
    # Manually drop a corrupt file and verify load returns a blank instead
    # of crashing the chat path.
    path = tmp_path / "relational" / "x.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{ this is not json", encoding="utf-8")
    m = await store.load("x")
    assert m.is_empty()
