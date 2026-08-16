"""Repeated reflections are penalised.

She kept restating one insight in fresh words — 腿动起来脑子能挂住 /
脚踩实地脑子没法飘 / 脚先落地眼先落一个实的东西 — and each copy landed at
importance 8, so a single worn theme crowded retrieval and fed the planner.

Lexical comparison cannot see this: those pairs score ~0.2, the same as
unrelated ones. Novelty is judged on embeddings instead.
"""
import pytest

from lingxi.facts.models import Fact, FactType, Source
from lingxi.facts.reflector import Reflector, _cosine
from datetime import datetime


class StubEmbedder:
    """Maps text to a vector by theme, mimicking the measured similarities."""
    THEMES = {"身体": [1.0, 0.0, 0.0], "留白": [0.0, 1.0, 0.0], "家人": [0.0, 0.0, 1.0]}

    async def embed(self, text: str):
        for theme, vec in self.THEMES.items():
            if theme in text:
                return vec
        return [0.4, 0.4, 0.4]


def _pattern(content: str) -> Fact:
    return Fact(subject="aria", content=content, source=Source.LLM_INFERRED,
                type=FactType.PATTERN, ts=datetime.now(), importance=8)


class StubStore:
    def __init__(self, patterns): self._p = patterns
    async def query(self, **kw): return self._p


class StubRetriever:
    def __init__(self, patterns): self._store = StubStore(patterns)


def _reflector(patterns, **kw):
    return Reflector(llm=None, retriever=StubRetriever(patterns),
                     inference_writer=None, embedder=StubEmbedder(), **kw)


def test_cosine_basics():
    assert _cosine([1, 0], [1, 0]) == pytest.approx(1.0)
    assert _cosine([1, 0], [0, 1]) == pytest.approx(0.0)
    assert _cosine([], [1]) == 0.0          # mismatched/empty is not an error


@pytest.mark.asyncio
async def test_restatement_is_dropped():
    r = _reflector([_pattern("身体 的洞见")])
    assert await r._novelty("身体 换个说法") is None


@pytest.mark.asyncio
async def test_echoes_lower_importance_one_step_each():
    r = _reflector([_pattern("身体 一"), _pattern("身体 二"), _pattern("家人 三")],
                   restate_threshold=1.1)   # never drop, only count
    echoes, closest = await r._novelty("身体 四")
    assert echoes == 2                      # the two 身体 ones, not 家人
    assert max(3, 8 - echoes) == 6


@pytest.mark.asyncio
async def test_distinct_insight_keeps_full_importance():
    r = _reflector([_pattern("身体 一"), _pattern("留白 二")])
    echoes, _ = await r._novelty("家人 三")
    assert echoes == 0
    assert max(3, 8 - echoes) == 8


@pytest.mark.asyncio
async def test_importance_has_a_floor():
    r = _reflector([_pattern(f"身体 {i}") for i in range(9)], restate_threshold=1.1)
    echoes, _ = await r._novelty("身体 又一条")
    assert max(3, 8 - echoes) == 3          # worn thin, never zero


@pytest.mark.asyncio
async def test_missing_embedder_never_blocks_reflection():
    r = Reflector(llm=None, retriever=StubRetriever([_pattern("身体 一")]),
                  inference_writer=None, embedder=None)
    assert await r._novelty("任何内容") == (0, 0.0)
