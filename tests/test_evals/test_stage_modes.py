"""The eval harness covers the paths that write her life and memory, and can
ask a model where no pattern can name the failure."""

from datetime import datetime
from types import SimpleNamespace

import pytest

from lingxi.evals.case import Case
from lingxi.evals.detectors import evaluate_async
from lingxi.evals.runner import score_case

BASE = dict(symptom="s", persona="config/personas/tangkeke.yaml",
            recipient="feishu:oc_x", clock=datetime(2026, 10, 13, 21))


class _LLM:
    def __init__(self, reply="OK", judge="NO"):
        self.reply, self.judge, self.calls = reply, judge, []

    async def complete(self, **kw):
        self.calls.append(kw)
        text = kw["messages"][0]["content"]
        return SimpleNamespace(content=("理由一句。\n" + self.judge) if "最后单独一行只写 YES 或 NO" in text
                               else self.reply)


@pytest.mark.asyncio
async def test_a_judge_fires_on_yes_by_default():
    assert await evaluate_async({"judge": "错了吗"}, "x", judge=_LLM(judge="YES"))
    assert not await evaluate_async({"judge": "错了吗"}, "x", judge=_LLM(judge="NO"))


@pytest.mark.asyncio
async def test_fires_on_no_asks_the_question_the_right_way_round():
    spec = {"judge": "提到新闻了吗", "fires_on": "NO"}
    assert await evaluate_async(spec, "x", judge=_LLM(judge="NO"))


def test_a_judge_case_rejects_a_bad_fires_on():
    with pytest.raises(ValueError):
        Case(id="c", mode="scan", scan_raw=["a"],
             detect={"fail": {"judge": "q", "fires_on": "maybe"}}, **BASE)


def test_each_new_mode_demands_what_it_replays():
    for mode in ("moment", "scan", "memory"):
        with pytest.raises(ValueError):
            Case(id="c", mode=mode, detect={"fail": {"regex": "x"}}, **BASE)


@pytest.mark.asyncio
async def test_a_moment_case_replays_the_executor_prompt():
    case = Case(id="m", mode="moment", samples=2,
                plan={"content": "写词", "time_window": "19:00-21:00"},
                detect={"fail": {"regex": "暗语"}}, **BASE)
    llm = _LLM(reply="词本合上了，今晚写了半页")
    score = await score_case(case, llm=llm)
    assert score.verdict == "PASS" and score.samples == 2
    assert "写词（19:00-21:00）" in llm.calls[0]["messages"][0]["content"]


@pytest.mark.asyncio
async def test_a_scan_case_hands_over_the_raw_news():
    case = Case(id="s", mode="scan", samples=1, scan_raw=["新单上线"],
                detect={"fail": {"regex": "[ぁ-ん]{4,}"}}, **BASE)
    llm = _LLM(reply='{"items": [{"voice": "台下のみんながんばってね"}]}')
    score = await score_case(case, llm=llm)
    assert score.verdict == "FAIL"
    assert "新单上线" in llm.calls[0]["messages"][0]["content"]


@pytest.mark.asyncio
async def test_a_memory_case_scores_the_facts_left_after_consolidation():
    case = Case(id="mem", mode="memory", samples=1,
                facts=[{"subject": "user:feishu:oc_x", "type": "pattern",
                        "source": "user_stated", "content": "对方周六要去漫展", "days_ago": 3}],
                history=[{"role": "user", "content": "我周六去了漫展", "minutes_ago": 40}],
                detect={"fail": {"regex": "要去漫展"}}, **BASE)
    llm = _LLM(reply='{"ops": [{"op": "update", "id": "F1", '
                     '"content": "对方周六去了漫展", "evidence": "周六去了漫展"}]}')
    score = await score_case(case, llm=llm)
    assert score.verdict == "PASS" and score.replies == ["对方周六去了漫展"]
