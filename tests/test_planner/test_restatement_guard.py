"""A plan step with nothing new in it must not fill the hour with rewordings.

A step spans a couple of hours and this ticks every thirty minutes, so when
nothing has moved the model restates. Measured over three days of her own
events: 「守着锅等它咕嘟，旋律已经出去了」 then 「守着锅等粥咕嘟，旋律刚才已经
跑出去了」 then 「守着粥等它最后咕嘟一下」 — two hours of waiting for porridge,
four entries, one moment.
"""

from datetime import datetime, timedelta

import pytest

from lingxi.facts.models import Fact, FactType, Source
from lingxi.planner.executor import PlanExecutor


PREV = "守着锅等它咕嘟，旋律已经出去了，现在就差这锅粥了。"
SAME = "守着锅等粥咕嘟，旋律刚才已经跑出去了，现在就差这最后一步。"
MOVED = "粥喝完了，碗搁桌上，进镜子。"


class _LLM:
    """Returns each queued answer in turn; records the prompts."""

    def __init__(self, *answers):
        self._answers = list(answers)
        self.prompts = []

    async def complete(self, **kwargs):
        self.prompts.append(kwargs["messages"][0]["content"])
        text = self._answers.pop(0) if self._answers else ""
        return type("R", (), {"content": text})()


class _Embedder:
    """SAME is near PREV; MOVED is orthogonal to both."""

    VECTORS = {PREV: [1.0, 0.0], SAME: [0.99, 0.14], MOVED: [0.0, 1.0]}

    async def embed(self, text):
        return self.VECTORS.get(text, [0.0, 1.0])


class _Boom:
    async def embed(self, text):
        raise RuntimeError("embedding endpoint down")


class _Retriever:
    def __init__(self, previous):
        self._previous = previous

    async def fetch(self, query):
        if not self._previous:
            return []
        return [Fact(subject="aria", content=self._previous,
                     source=Source.LIFE_SIMULATED, type=FactType.EVENT,
                     ts=datetime.now() - timedelta(minutes=30))]


class _Writer:
    def __init__(self):
        self.written = []

    async def write(self, fact):
        self.written.append(fact.content)


def _executor(llm, writer, *, previous=PREV, embedder=None):
    ex = object.__new__(PlanExecutor)
    ex._llm = llm
    ex._retriever = _Retriever(previous)
    ex._writer = writer
    ex._planner = None
    ex._model = None
    ex._embedder = embedder if embedder is not None else _Embedder()
    ex._replan_requested = False
    ex._self_ctx = "你是她。"
    ex._find_current_plan = lambda now: _plan()
    return ex


async def _plan():
    return Fact(subject="aria", content="早饭 + 开嗓", source=Source.LIFE_SIMULATED,
                type=FactType.PLAN, ts=datetime.now(), tags=["time_window:07:00-09:00"])


@pytest.mark.asyncio
async def test_a_restatement_gets_one_nudge_and_then_lands():
    llm = _LLM(SAME, MOVED)
    writer = _Writer()

    await _executor(llm, writer).tick()

    assert writer.written == [MOVED]
    assert "接下来发生的" in llm.prompts[1], "the retry has to say what to do"


@pytest.mark.asyncio
async def test_a_second_restatement_writes_nothing():
    """An hour with one entry is honest; four about the porridge is not."""
    llm = _LLM(SAME, SAME)
    writer = _Writer()

    await _executor(llm, writer).tick()

    assert writer.written == []


@pytest.mark.asyncio
async def test_a_moment_that_moved_is_written_straight_away():
    llm = _LLM(MOVED)
    writer = _Writer()

    await _executor(llm, writer).tick()

    assert writer.written == [MOVED]
    assert len(llm.prompts) == 1, "no second call when the first one moved"


@pytest.mark.asyncio
async def test_the_first_moment_of_the_day_has_nothing_to_repeat():
    llm = _LLM(SAME)
    writer = _Writer()

    await _executor(llm, writer, previous="").tick()

    assert writer.written == [SAME]


@pytest.mark.asyncio
async def test_a_broken_embedder_still_writes():
    """A repeat costs a line; refusing to write costs the hour."""
    llm = _LLM(SAME)
    writer = _Writer()

    await _executor(llm, writer, embedder=_Boom()).tick()

    assert writer.written == [SAME]


@pytest.mark.asyncio
async def test_no_embedder_keeps_the_old_behaviour():
    llm = _LLM(SAME)
    writer = _Writer()

    await _executor(llm, writer, embedder=False).tick()

    assert writer.written == [SAME]


@pytest.mark.asyncio
async def test_an_empty_generation_writes_nothing():
    writer = _Writer()
    await _executor(_LLM(""), writer).tick()
    assert writer.written == []


@pytest.mark.asyncio
async def test_the_prompt_asks_for_what_changed():
    llm = _LLM(MOVED)
    await _executor(llm, _Writer()).tick()

    assert "新发生的" in llm.prompts[0]


# --- where in the step we are -----------------------------------------

class TestPosition:
    """Steps run 30 to 210 minutes while this ticks every 30.

    A 3.5-hour step has to yield seven distinct moments. Given only a clock
    reading the model restated the start of the block for two hours — and the
    restatement guard would then throw those ticks away, emptying a morning.
    """

    def _at(self, hhmm, window):
        from lingxi.planner.executor import _parse_time_window, describe_position
        h, m = map(int, hhmm.split(":"))
        return describe_position(h * 60 + m, *_parse_time_window(window))

    def test_the_start_of_a_long_step(self):
        assert "刚开始" in self._at("08:50", "08:30-12:00")

    def test_the_middle_says_how_much_is_left(self):
        assert "还剩 100 分钟" in self._at("10:20", "08:30-12:00")

    def test_the_end_asks_to_wrap_up(self):
        assert "收尾" in self._at("11:40", "08:30-12:00")

    def test_a_short_step_is_not_carved_up(self):
        """A 30-minute step gets one tick; beginning/middle/end is nonsense."""
        assert self._at("07:10", "07:00-07:30") == "这一段就这么点时间"

    def test_a_step_crossing_midnight(self):
        assert "收尾" in self._at("00:45", "23:00-01:00")

    def test_the_three_phases_of_one_step_differ(self):
        w = "08:30-12:00"
        assert len({self._at("08:40", w), self._at("10:20", w),
                    self._at("11:50", w)}) == 3
