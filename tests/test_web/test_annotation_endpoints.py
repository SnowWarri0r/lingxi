"""Tests for the web annotation endpoints.

Uses FastAPI TestClient — requires the 'api' extra installed.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from lingxi.fewshot.models import AnnotationTurn
from lingxi.fewshot.store import AnnotationStore, FewShotStore
from lingxi.web.server import create_app


class FakeMemory:
    def __init__(self, embedding_provider):
        self.embedding_provider = embedding_provider


class FakeEngine:
    def __init__(self, ann_store, few_store, llm):
        self.annotation_store = ann_store
        self.fewshot_store = few_store
        self.llm = llm
        self.memory = FakeMemory(embedding_provider=llm)
        self.fewshot_retriever = None


class FakeLLM:
    async def embed(self, text):
        return [0.0] * 16

    async def complete(self, messages, **kwargs):
        from lingxi.providers.base import CompletionResult
        return CompletionResult(content="场景：测试\n标签：t1,t2")


@pytest.fixture
async def app_and_store(tmp_path):
    ann = AnnotationStore(data_dir=tmp_path)
    pool = FewShotStore(data_dir=tmp_path, embedding_dim=16)
    await pool.init()
    await ann.record(AnnotationTurn(
        turn_id="t-web-1", recipient_key="web:user",
        user_message="hi", inner_thought="thinking",
        speech="hello",
    ))
    engine = FakeEngine(ann, pool, FakeLLM())
    app = create_app(engine=engine)
    return app, ann, pool


def test_post_annotate_positive(app_and_store):
    app, ann, pool = app_and_store
    client = TestClient(app)
    resp = client.post(
        "/turns/t-web-1/annotate",
        json={"kind": "positive"},
    )
    assert resp.status_code == 200


def test_post_annotate_correction(app_and_store):
    app, ann, pool = app_and_store
    client = TestClient(app)
    resp = client.post(
        "/turns/t-web-1/annotate",
        json={"kind": "negative", "correction": "嗨"},
    )
    assert resp.status_code == 200


def test_get_inner_thought(app_and_store):
    app, ann, pool = app_and_store
    client = TestClient(app)
    resp = client.get("/turns/t-web-1/inner_thought")
    assert resp.status_code == 200
    assert resp.json()["inner_thought"] == "thinking"


def test_get_inner_thought_missing(app_and_store):
    app, ann, pool = app_and_store
    client = TestClient(app)
    resp = client.get("/turns/nope/inner_thought")
    assert resp.status_code == 404
