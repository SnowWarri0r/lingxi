"""Tests for the extractor's parsing + merge logic."""

import json
from datetime import datetime
from types import SimpleNamespace

import pytest

from lingxi.providers.base import CompletionResult
from lingxi.relational.extractor import (
    extract_relational_deltas,
    merge_deltas_into_memory,
)
from lingxi.relational.models import RelationalMemory


class _FakeLLM:
    """Minimal LLMProvider stub returning a fixed response."""

    def __init__(self, payload: str):
        self._payload = payload

    async def complete(self, **_kwargs):
        return CompletionResult(content=self._payload)


@pytest.mark.asyncio
async def test_clean_json_response_parses():
    llm = _FakeLLM(json.dumps({
        "inside_jokes": [{"phrase": "蜘蛛会做梦的", "origin": "梦"}],
        "shared_places": [],
        "fight_patterns": [],
        "sweet_moments": [],
        "pet_names": ["笨蛋"],
        "daily_patterns": [],
        "relationship_summary_update": None,
    }))
    deltas = await extract_relational_deltas(
        llm, RelationalMemory(recipient_key="x"), turns=[]
    )
    assert deltas["inside_jokes"][0]["phrase"] == "蜘蛛会做梦的"
    assert deltas["pet_names"] == ["笨蛋"]


@pytest.mark.asyncio
async def test_markdown_fence_tolerated():
    payload = "```json\n" + json.dumps({
        "inside_jokes": [],
        "shared_places": [],
        "fight_patterns": [],
        "sweet_moments": [],
        "pet_names": ["X"],
        "daily_patterns": [],
        "relationship_summary_update": None,
    }) + "\n```"
    llm = _FakeLLM(payload)
    deltas = await extract_relational_deltas(
        llm, RelationalMemory(recipient_key="x"), turns=[]
    )
    assert deltas["pet_names"] == ["X"]


@pytest.mark.asyncio
async def test_garbage_response_returns_empty():
    llm = _FakeLLM("not json at all")
    deltas = await extract_relational_deltas(
        llm, RelationalMemory(recipient_key="x"), turns=[]
    )
    # All buckets default-empty
    assert deltas["inside_jokes"] == []
    assert deltas["pet_names"] == []
    assert deltas["relationship_summary_update"] is None


class TestMergeDeltas:
    def _payload(self, **overrides):
        base = {
            "inside_jokes": [],
            "shared_places": [],
            "fight_patterns": [],
            "sweet_moments": [],
            "pet_names": [],
            "daily_patterns": [],
            "relationship_summary_update": None,
        }
        base.update(overrides)
        return base

    def test_merge_adds_inside_joke(self):
        m = RelationalMemory(recipient_key="x")
        added = merge_deltas_into_memory(
            m, self._payload(inside_jokes=[{"phrase": "X", "origin": "Y"}])
        )
        assert added == 1
        assert m.inside_jokes[0].phrase == "X"

    def test_merge_dedups_existing_phrase(self):
        from lingxi.relational.models import InsideJoke
        m = RelationalMemory(
            recipient_key="x",
            inside_jokes=[InsideJoke(phrase="X", origin="old")],
        )
        added = merge_deltas_into_memory(
            m, self._payload(inside_jokes=[{"phrase": "X", "origin": "new"}])
        )
        assert added == 0
        # Original origin preserved (not overwritten on dedup)
        assert m.inside_jokes[0].origin == "old"

    def test_merge_respects_existing_pet_names(self):
        m = RelationalMemory(recipient_key="x", pet_names=["A"])
        merge_deltas_into_memory(m, self._payload(pet_names=["A", "B"]))
        assert sorted(m.pet_names) == ["A", "B"]

    def test_merge_invalid_weight_falls_to_medium(self):
        m = RelationalMemory(recipient_key="x")
        merge_deltas_into_memory(
            m,
            self._payload(
                sweet_moments=[{"content": "X", "weight": "garbage"}],
            ),
        )
        assert m.sweet_moments[0].weight == "medium"

    def test_merge_summary_update_overwrites(self):
        m = RelationalMemory(recipient_key="x", relationship_summary="old")
        merge_deltas_into_memory(
            m, self._payload(relationship_summary_update="new")
        )
        assert m.relationship_summary == "new"

    def test_merge_summary_none_keeps_old(self):
        m = RelationalMemory(recipient_key="x", relationship_summary="old")
        merge_deltas_into_memory(m, self._payload(relationship_summary_update=None))
        assert m.relationship_summary == "old"

    def test_merge_skips_malformed_dict(self):
        m = RelationalMemory(recipient_key="x")
        merge_deltas_into_memory(
            m,
            self._payload(
                inside_jokes=["not a dict", {"phrase": "ok", "origin": "y"}],
            ),
        )
        assert len(m.inside_jokes) == 1
        assert m.inside_jokes[0].phrase == "ok"

    def test_merge_sets_last_extracted_at(self):
        m = RelationalMemory(recipient_key="x")
        assert m.last_extracted_at is None
        merge_deltas_into_memory(m, self._payload())
        assert m.last_extracted_at is not None

    def test_merge_adds_signature_phrases(self):
        m = RelationalMemory(recipient_key="x")
        merge_deltas_into_memory(
            m, self._payload(signature_phrases=["懂", "等下"])
        )
        assert "懂" in m.signature_phrases
        assert "等下" in m.signature_phrases

    def test_merge_dedupes_existing_signature_phrases(self):
        m = RelationalMemory(recipient_key="x", signature_phrases=["懂"])
        added = merge_deltas_into_memory(
            m, self._payload(signature_phrases=["懂", "等下"])
        )
        # "懂" deduped, "等下" added
        assert added == 1
        assert m.signature_phrases.count("懂") == 1
        assert "等下" in m.signature_phrases

    def test_merge_skips_non_string_signature_entry(self):
        m = RelationalMemory(recipient_key="x")
        merge_deltas_into_memory(
            m, self._payload(signature_phrases=["ok", {"phrase": "x"}, None])
        )
        assert m.signature_phrases == ["ok"]
