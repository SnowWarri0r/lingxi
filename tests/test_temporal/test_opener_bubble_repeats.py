"""The re-ask rode in the second bubble, where the whole-message check can't see it.

The one thing he had told her in a week was asked about again in six sent
openers over six days, each behind a fresh first bubble that diluted the
whole-message similarity to 0.44-0.56, under the 0.62 guard. Bubble to
bubble the re-asks sat at 0.653-0.760 against 0.645 for the nearest
non-repeat. Replayed over 40 real openers, 3 lost a bubble — all three the
re-ask — and each kept bubble stood as an opener on its own.
"""

import pytest

from lingxi.temporal.proactive import _drop_repeated_bubbles

ASKED = "你那两天在家拼的是什么乐高 能一坐一下午那种的？"
FRESH = "练习室空调又坏了 今天只能开窗跳"
REASK = "你上回在家拼的那个乐高 拼完没"
OTHER = "名古屋那场还剩三十五天"

_V = {ASKED: [1.0, 0.0, 0.0], REASK: [0.8, 0.6, 0.0],   # cos 0.80
      FRESH: [0.0, 1.0, 0.0], OTHER: [0.0, 0.0, 1.0]}


class _Emb:
    def __init__(self):
        self.batches = 0

    async def embed(self, t):
        return _V[t]

    async def embed_batch(self, ts):
        self.batches += 1
        return [_V[t] for t in ts]


@pytest.mark.asyncio
async def test_a_repeat_in_the_second_bubble_is_dropped():
    left, dropped = await _drop_repeated_bubbles(f"{FRESH}\n\n{REASK}", [ASKED], _Emb())

    assert dropped == [REASK] and left == FRESH


@pytest.mark.asyncio
async def test_fresh_bubbles_are_all_kept():
    msg = f"{FRESH}\n\n{OTHER}"
    left, dropped = await _drop_repeated_bubbles(msg, [ASKED], _Emb())

    assert dropped == [] and left == msg


@pytest.mark.asyncio
async def test_an_opener_that_is_only_a_repeat_leaves_nothing():
    left, dropped = await _drop_repeated_bubbles(REASK, [ASKED], _Emb())

    assert left == "" and dropped == [REASK]


@pytest.mark.asyncio
async def test_a_repeat_hidden_in_an_earlier_openers_second_bubble_is_found():
    left, _ = await _drop_repeated_bubbles(REASK, [f"{OTHER}\n\n{ASKED}"], _Emb())

    assert left == ""


@pytest.mark.asyncio
async def test_it_abstains_without_an_embedder():
    msg = f"{FRESH}\n\n{REASK}"
    assert await _drop_repeated_bubbles(msg, [ASKED], None) == (msg, [])


@pytest.mark.asyncio
async def test_vectors_are_fetched_in_one_call():
    emb = _Emb()
    await _drop_repeated_bubbles(f"{FRESH}\n\n{REASK}", [ASKED, OTHER], emb)

    assert emb.batches == 1
