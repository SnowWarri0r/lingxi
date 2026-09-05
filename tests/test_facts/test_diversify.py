"""Eight slots should hold eight subjects, not the top eight of one.

Asked for the eight best facts about him on 2026-08-31, the opener block got
the Chengdu autograph three times, the handwritten letter twice and the trip
twice — three subjects, out of twenty-eight distinct facts she held. Freeing
those slots surfaced how he defines the friendship, which had been stored for
weeks and never once shown to her.
"""

import pytest

from lingxi.facts.diversify import select_diverse


class _Fact:
    def __init__(self, content):
        self.content = content

    def __repr__(self):
        return f"<{self.content}>"


class _Embedder:
    """Vectors by exact text; anything unregistered is orthogonal to all."""

    def __init__(self, vectors):
        self._v = vectors
        self.calls = 0

    async def embed(self, text):
        self.calls += 1
        return self._v.get(text, [0.0, 0.0, 1.0])


class _Boom:
    async def embed(self, text):
        raise RuntimeError("embedding endpoint down")


# Three restatements of one subject, then two other subjects.
A1, A2, A3 = _Fact("亲签a"), _Fact("亲签b"), _Fact("亲签c")
B, C = _Fact("手写信"), _Fact("闺蜜")
EMB = _Embedder({
    "亲签a": [1.0, 0.0, 0.0], "亲签b": [0.99, 0.1, 0.0], "亲签c": [0.98, 0.15, 0.0],
    "手写信": [0.0, 1.0, 0.0], "闺蜜": [0.0, 0.0, 1.0],
})


@pytest.mark.asyncio
async def test_one_subject_does_not_take_every_slot():
    picked = await select_diverse([A1, A2, A3, B, C], 3, EMB)

    assert picked == [A1, B, C]


@pytest.mark.asyncio
async def test_the_best_fact_is_always_kept():
    """Input order is the caller's ranking; the top one never loses."""
    picked = await select_diverse([A1, A2, A3, B, C], 2, EMB)

    assert picked[0] is A1


@pytest.mark.asyncio
async def test_ranking_order_survives():
    picked = await select_diverse([A1, A2, A3, B, C], 3, EMB)

    assert picked == sorted(picked, key=lambda f: [A1, A2, A3, B, C].index(f))


@pytest.mark.asyncio
async def test_a_short_pool_is_topped_up_rather_than_left_half_empty():
    """Skipping must not hand back three facts when eight were asked for."""
    picked = await select_diverse([A1, A2, A3], 3, EMB)

    assert len(picked) == 3, "a repetitive block beats a half-empty one"


@pytest.mark.asyncio
async def test_top_ups_keep_ranking_order():
    picked = await select_diverse([A1, A2, A3], 2, EMB)

    assert picked == [A1, A2]


@pytest.mark.asyncio
async def test_distinct_facts_are_all_kept():
    picked = await select_diverse([A1, B, C], 3, EMB)

    assert picked == [A1, B, C]


@pytest.mark.asyncio
async def test_a_broken_embedder_degrades_to_the_old_behaviour():
    """Losing variety is survivable; raising here would cost the turn."""
    picked = await select_diverse([A1, A2, A3, B, C], 3, _Boom())

    assert picked == [A1, A2, A3]


@pytest.mark.asyncio
async def test_no_embedder_degrades_to_the_old_behaviour():
    assert await select_diverse([A1, A2, A3], 2, None) == [A1, A2]


@pytest.mark.asyncio
async def test_an_empty_pool_is_not_an_error():
    assert await select_diverse([], 8, EMB) == []


@pytest.mark.asyncio
async def test_asking_for_nothing_costs_no_embedding():
    fresh = _Embedder({})
    assert await select_diverse([A1, B], 0, fresh) == []
    assert fresh.calls == 0


@pytest.mark.asyncio
async def test_a_single_fact_costs_no_embedding():
    fresh = _Embedder({})
    assert await select_diverse([A1], 8, fresh) == [A1]
    assert fresh.calls == 0
