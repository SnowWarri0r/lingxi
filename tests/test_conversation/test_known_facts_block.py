"""The block that exists to prevent duplicates was itself mostly duplicates.

【已经记住的关于对方的事】 is shown to the orchestrator so item 8 can follow
its own rule — 同一件事换个说法不用再写一遍. Ranked by score alone, the fifteen
slots carried nine facts on 09-21: the handwritten letter three times, the
autograph twice, the trip twice, 追了好几场都没抽中 twice. Eleven of the fifteen
described one afternoon a month past, and the facts pushed out were the ones
still true of him — where he lives, when he finishes work, how he commutes,
what he does at lunch, and a trip nine days away at the time.

Write-side dedup does not reach this and is not meant to: those are distinct
sentences, and most predate the guard. Assembly is where it gets fixed.

Similarities measured on the live store with the production embedder: the
genuine restatements sat at 0.918, 0.869, 0.737, 0.720, 0.716 and 0.708, and
the nearest pair that says two different things at 0.616 — either side of the
0.62 select_diverse already uses elsewhere.
"""

import pytest

from lingxi.facts.diversify import select_diverse


class Fact:
    def __init__(self, content, vec):
        self.content = content
        self.vec = vec


class Embedder:
    """Stands in for the production embedder, reproducing the measured gaps."""

    async def embed(self, text):
        return _VECS[text]

    async def embed_batch(self, texts):
        return [_VECS[t] for t in texts]


# Three restatements of the letter, two of the trip, then distinct facts.
LETTER_A = "对方周六去邻市看阿澪的专场，两场都去，准备了一封信要当面给"
LETTER_B = "对方给阿澪写了封信，信封上贴了小熊贴纸，周六见面时给"
LETTER_C = "对方准备了一封信给阿澪，贴了小熊贴纸，打算周六见面时交给她"
TRIP_A = "对方这周六要去参加阿澪的见面会，能当面说两次话"
TRIP_B = "对方周六去邻市见到了阿澪本人，抽到了签名照"
HOME = "对方住在邻市"
SEVEN = "对方一般晚上七点半下班"
COMMUTE = "对方平时坐地铁上下班"
NEXT_TRIP = "对方说下个月会去漫展见阿澪"

# One axis per subject; restatements sit close along their own axis, and the
# ~0.3 between 七点半下班 and 坐地铁上下班 is the real gap between two facts
# that are both about his commute and are not the same fact.
_VECS = {
    LETTER_A:  [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    LETTER_B:  [0.95, 0.1, 0.0, 0.0, 0.0, 0.05],
    LETTER_C:  [0.9, 0.0, 0.1, 0.0, 0.0, 0.0],
    TRIP_A:    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    TRIP_B:    [0.1, 0.95, 0.0, 0.0, 0.0, 0.0],
    HOME:  [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    SEVEN:      [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    COMMUTE:   [0.0, 0.0, 0.0, 0.3, 0.9, 0.0],
    NEXT_TRIP: [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
}

# Six distinct things are said across the nine lines, so six is the block that
# holds them all — the width the restatements were costing.
BLOCK = 6

# Score order as the retriever returns it: the month-old trip outranks
# everything that is merely still true.
RANKED = [Fact(c, _VECS[c]) for c in
          (LETTER_A, LETTER_B, LETTER_C, TRIP_A, TRIP_B,
           HOME, SEVEN, COMMUTE, NEXT_TRIP)]


@pytest.mark.asyncio
async def test_one_slot_per_thing_not_three():
    picked = await select_diverse(RANKED, BLOCK, Embedder())
    said = [f.content for f in picked]

    assert LETTER_A in said
    assert LETTER_B not in said and LETTER_C not in said


@pytest.mark.asyncio
async def test_what_is_still_true_of_him_gets_in():
    """The slots the restatements were occupying."""
    said = [f.content for f in await select_diverse(RANKED, BLOCK, Embedder())]

    assert HOME in said and SEVEN in said


@pytest.mark.asyncio
async def test_next_weeks_plan_is_not_crowded_out_by_last_months():
    said = [f.content for f in await select_diverse(RANKED, BLOCK, Embedder())]

    assert NEXT_TRIP in said


@pytest.mark.asyncio
async def test_the_best_ranked_fact_is_never_the_one_dropped():
    picked = await select_diverse(RANKED, 3, Embedder())

    assert picked[0].content == LETTER_A


@pytest.mark.asyncio
async def test_a_full_block_is_still_a_full_block():
    """Skipping must not shrink it — a thin block is worse than a repetitive
    one, so what was passed over tops it back up."""
    picked = await select_diverse(RANKED, 8, Embedder())

    assert len(picked) == 8


class TestItAsksForVectorsOnce:
    """The reply path waits on this. Serial round-trips cost 6.6s for 30
    facts against 0.5s batched, measured on the production embedder."""

    @pytest.mark.asyncio
    async def test_one_batched_call_not_one_per_fact(self):
        calls = {"batch": 0, "single": 0}

        class Counting(Embedder):
            async def embed(self, text):
                calls["single"] += 1
                return _VECS[text]

            async def embed_batch(self, texts):
                calls["batch"] += 1
                return [_VECS[t] for t in texts]

        await select_diverse(RANKED, BLOCK, Counting())

        assert calls["batch"] == 1 and calls["single"] == 0

    @pytest.mark.asyncio
    async def test_an_embedder_offering_only_embed_still_diversifies(self):
        """Batching must not become a requirement — losing diversity to gain
        speed would trade the fix for the optimisation."""
        class SingleOnly:
            async def embed(self, text):
                return _VECS[text]

        said = [f.content for f in await select_diverse(RANKED, BLOCK, SingleOnly())]

        assert LETTER_B not in said and NEXT_TRIP in said

    @pytest.mark.asyncio
    async def test_a_failing_embedder_still_yields_a_block(self):
        class Boom:
            async def embed_batch(self, texts):
                raise RuntimeError("no")

        picked = await select_diverse(RANKED, 4, Boom())

        assert [f.content for f in picked] == [f.content for f in RANKED[:4]]
