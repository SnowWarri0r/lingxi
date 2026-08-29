"""The same question in different words is still the same question.

Delivered three days apart, both to the same person:

    你那边最近忙完了吗 都好几天没听到你消息了
    都一个星期没你消息了 你最近是不是特别忙呀

The character-level guard compares them at 0.30 and needs 0.75, so both went
out. Four of the nineteen messages she has ever sent were variants of this
one, which is the thing that reads as mechanical.
"""

import pytest

from lingxi.temporal.proactive import (
    _SEMANTIC_DUP_THRESHOLD,
    _cosine,
    _semantically_too_similar,
    _too_similar,
)


class _Embedder:
    """Returns the vector registered for a text, or an orthogonal one."""

    def __init__(self, vectors: dict[str, list[float]]):
        self._vectors = vectors
        self.calls = 0

    async def embed(self, text: str) -> list[float]:
        self.calls += 1
        return self._vectors.get(text, [0.0, 0.0, 1.0])


class _Boom:
    async def embed(self, text: str):
        raise RuntimeError("embedding endpoint down")


A = "你那边最近忙完了吗 都好几天没听到你消息了"
B = "都一个星期没你消息了 你最近是不是特别忙呀"
UNRELATED = "今天彩排站在台上往下看 想着后天这里要坐满人 突然有点想哭"

# A and B point almost the same way; the unrelated message is orthogonal.
NEAR = _Embedder({A: [1.0, 0.0, 0.0], B: [0.95, 0.3, 0.0],
                  UNRELATED: [0.0, 1.0, 0.0]})


def test_the_character_guard_cannot_see_this():
    """Establishes why the semantic check exists, not just that it works."""
    assert _too_similar(B, [A]) is None


@pytest.mark.asyncio
async def test_the_same_question_in_other_words_is_caught():
    assert await _semantically_too_similar(B, [A], NEAR) == A


@pytest.mark.asyncio
async def test_a_different_message_goes_through():
    assert await _semantically_too_similar(UNRELATED, [A], NEAR) is None


@pytest.mark.asyncio
async def test_the_repeat_it_names_is_the_one_it_matched():
    """The log line quotes it, so it has to be the right one."""
    embedder = _Embedder({A: [1.0, 0.0, 0.0], B: [0.95, 0.3, 0.0],
                          UNRELATED: [0.0, 1.0, 0.0]})
    assert await _semantically_too_similar(B, [UNRELATED, A], embedder) == A


@pytest.mark.asyncio
async def test_a_broken_embedder_lets_the_message_through():
    """Falling silent is the worse failure — abstain, don't block."""
    assert await _semantically_too_similar(B, [A], _Boom()) is None


@pytest.mark.asyncio
async def test_no_embedder_abstains():
    assert await _semantically_too_similar(B, [A], None) is None


@pytest.mark.asyncio
async def test_nothing_sent_before_costs_no_call():
    fresh = _Embedder({})
    assert await _semantically_too_similar(B, [], fresh) is None
    assert fresh.calls == 0, "nothing to compare against — don't embed"


@pytest.mark.asyncio
async def test_it_stops_at_the_first_match_it_finds():
    """Ten previous messages must not mean ten embeddings after a hit."""
    fresh = _Embedder({A: [1.0, 0.0, 0.0], B: [0.95, 0.3, 0.0]})
    await _semantically_too_similar(B, [A, "别的话", "又一句"], fresh)

    assert fresh.calls == 2, "the candidate, then one previous — then stop"


@pytest.mark.asyncio
async def test_an_empty_candidate_is_not_a_duplicate():
    assert await _semantically_too_similar("   ", [A], NEAR) is None


def test_cosine_handles_a_zero_vector():
    """A degenerate embedding must not raise mid-decision."""
    assert _cosine([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_the_threshold_sits_above_the_corpus_median():
    """Calibrated, not guessed: median pairwise similarity was 0.385."""
    assert 0.54 < _SEMANTIC_DUP_THRESHOLD < 0.70
