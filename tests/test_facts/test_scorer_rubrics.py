"""Her own life and what she knows about him are scored different questions.

One rubric asked how much an event moved her. Applied to facts about him,
「他是阿澪的粉丝，很在意抽选」 scored 3 — the same as a haircut — and
「他住在邻市」 came 28th of 39, both outside the pool that reaches the prompt,
squeezed out by a cluster of dated facts about one weekend. Knowing a person
runs on durability, not impact.
"""

import pytest

from lingxi.facts.models import Fact, FactType, Source
from lingxi.facts.scorer import ImportanceScorer, _bucket_key, _prompt_for


def _fact(subject, content, fid="f1"):
    from datetime import datetime
    return Fact(id=fid, subject=subject, content=content,
                source=Source.USER_STATED, type=FactType.PATTERN,
                ts=datetime(2026, 9, 7))


class _LLM:
    def __init__(self):
        self.prompts = []

    async def complete(self, **kwargs):
        self.prompts.append(kwargs["messages"][0]["content"])
        return type("R", (), {"content": '[{"id":"f1","score":7}]'})()


# --- which rubric ------------------------------------------------------

def test_her_own_events_are_rated_by_how_much_they_moved_her():
    assert "对**我**来说有多重要" in _prompt_for("aria")


def test_facts_about_him_are_rated_by_what_still_explains_him():
    assert "了解他这个人" in _prompt_for("other")
    assert "下个月它还能不能" in _prompt_for("other")


def test_an_npc_keeps_the_first_person_rubric():
    """An NPC lives a life of her own; the impact question fits."""
    assert _prompt_for(_bucket_key("npc:香音")) == _prompt_for("aria")


def test_the_two_rubrics_are_not_the_same_text():
    assert _prompt_for("other") != _prompt_for("aria")


def test_durable_traits_outrank_dated_detail_in_the_user_rubric():
    """The ordering the old rubric inverted, stated outright."""
    p = _prompt_for("other")
    assert p.index("一次性细节") < p.index("他是谁")


# --- the scorer picks by subject --------------------------------------

@pytest.mark.asyncio
async def test_a_user_fact_gets_the_user_rubric():
    llm = _LLM()
    scorer = ImportanceScorer(llm, batch_size=1)

    await scorer.score_one(_fact("user:feishu:oc_x", "对方住在邻市"))

    assert "了解他这个人" in llm.prompts[0]


@pytest.mark.asyncio
async def test_her_own_fact_gets_her_rubric():
    llm = _LLM()
    scorer = ImportanceScorer(llm, batch_size=1)

    await scorer.score_one(_fact("aria", "今天排练磨了一段换气"))

    assert "对**我**来说有多重要" in llm.prompts[0]
    assert "了解他这个人" not in llm.prompts[0]


@pytest.mark.asyncio
async def test_a_failed_batch_still_falls_back_to_the_source_default():
    class _Boom:
        async def complete(self, **kwargs):
            raise RuntimeError("scorer down")

    scorer = ImportanceScorer(_Boom(), batch_size=1)
    score = await scorer.score_one(_fact("user:feishu:oc_x", "对方住在邻市"))

    assert score == 7  # DEFAULT_IMPORTANCE[USER_STATED]
