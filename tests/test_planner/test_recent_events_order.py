""""What just happened" has to mean the last thing that happened.

tick() pulled its recent events through FactRetriever.fetch, which ranks
0.5*recency + 0.3*(importance/10). Across a two-hour window the recency term
moves 0.010 end to end while importance moves 0.27 — 27x more — so importance
decided the order. Measured over 194 real ticks: on 47% the guard's
`previous` was not the latest moment, and on 57% the list handed to the model
was not in chronological order. On 3% the moment just written was not in the
list at all.

The damage is visible: 千砂那段还在耳朵边转/懒得动 landed three ticks running
while the guard was comparing each one against an older entry.
"""

from datetime import datetime

import pytest

from lingxi.facts.models import Fact, FactType, Source
from lingxi.planner.executor import PlanExecutor


NOW = datetime(2026, 9, 8, 11, 0)


def _ev(hhmm, content, importance):
    h, m = map(int, hhmm.split(":"))
    return Fact(subject="aria", content=content, source=Source.LIFE_SIMULATED,
                type=FactType.EVENT, ts=NOW.replace(hour=h, minute=m),
                importance=importance)


# A real window: the newest moment is the least important one.
WINDOW = [
    _ev("09:30", "推开练习室的门，香音已经在了", 3),
    _ev("10:00", "盯着香音的背，她唱到第二段换气那儿停了一下", 7),
    _ev("10:30", "手指点着换气记号，跟她说再来一遍", 6),
]


class _LLM:
    def __init__(self, *answers):
        self._answers = list(answers)
        self.prompts = []

    async def complete(self, **kwargs):
        self.prompts.append(kwargs["messages"][0]["content"])
        return type("R", (), {
            "content": self._answers.pop(0) if self._answers else "新的一刻。"})()


class _Retriever:
    """Returns the window in the score order the real retriever produces."""

    def __init__(self, facts):
        # importance-dominated: 10:00 (imp7) outranks 10:30 (imp6)
        self._facts = sorted(facts, key=lambda f: -(f.importance or 5))

    async def fetch(self, query):
        return list(self._facts)


class _Writer:
    def __init__(self):
        self.written = []

    async def write(self, fact):
        self.written.append(fact.content)


class _Embedder:
    """Identical text scores 1.0; any two different texts score 0.0."""

    def __init__(self):
        self._seen: dict[str, int] = {}

    async def embed(self, text):
        idx = self._seen.setdefault(text, len(self._seen))
        vec = [0.0] * 64
        vec[idx % 64] = 1.0
        return vec


def _executor(llm, writer, facts=WINDOW):
    ex = object.__new__(PlanExecutor)
    ex._llm = llm
    ex._retriever = _Retriever(facts)
    ex._writer = writer
    ex._planner = None
    ex._model = None
    ex._embedder = _Embedder()
    ex._replan_requested = False
    ex._self_ctx = "你是她。"

    async def _plan(now):
        return Fact(subject="aria", content="排练", source=Source.LIFE_SIMULATED,
                    type=FactType.PLAN, ts=NOW, tags=["time_window:09:00-12:00"])
    ex._find_current_plan = _plan
    return ex


def _listed(prompt):
    """The recent-events bullets, in the order the model reads them."""
    lines = [ln.strip().lstrip("- ") for ln in prompt.splitlines()
             if ln.strip().startswith("-")]
    return [ln for ln in lines if any(f.content[:8] in ln for f in WINDOW)]


@pytest.mark.asyncio
async def test_the_model_reads_the_moments_in_the_order_they_happened():
    llm = _LLM("新的一刻。")
    await _executor(llm, _Writer()).tick()

    listed = _listed(llm.prompts[0])
    assert len(listed) == len(WINDOW), "every recent moment should be listed"
    order = []
    for line in listed:
        for i, f in enumerate(WINDOW):
            if f.content[:8] in line:
                order.append(i)
                break
    assert order == [0, 1, 2], f"out of chronological order: {order}"


@pytest.mark.asyncio
async def test_the_guard_compares_against_the_latest_moment():
    """The newest entry is the least important one; it is still the previous."""
    llm = _LLM("手指点着换气记号，跟她说再来一遍", "真的往前走了一步。")
    writer = _Writer()

    await _executor(llm, writer).tick()

    assert len(llm.prompts) == 2, (
        "restating the 10:30 moment should have triggered the nudge")
    assert writer.written == ["真的往前走了一步。"]


@pytest.mark.asyncio
async def test_a_moment_that_moved_is_written_without_a_retry():
    llm = _LLM("走出练习室，天已经黑了。")
    writer = _Writer()

    await _executor(llm, writer).tick()

    assert writer.written == ["走出练习室，天已经黑了。"]
    assert len(llm.prompts) == 1


@pytest.mark.asyncio
async def test_a_single_event_window_still_works():
    llm = _LLM("接着来。")
    writer = _Writer()

    await _executor(llm, writer, facts=[WINDOW[0]]).tick()

    assert writer.written == ["接着来。"]


@pytest.mark.asyncio
async def test_an_empty_window_still_writes():
    llm = _LLM("今天第一件事。")
    writer = _Writer()

    await _executor(llm, writer, facts=[]).tick()

    assert writer.written == ["今天第一件事。"]
