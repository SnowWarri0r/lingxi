"""A case proves its detector before it scores, and baselines accumulate.

On 10-22 three of four judged cases marked the production message they were
built from as fine. A case carrying controls is BROKEN until its detector
gets them right.
"""

from datetime import datetime
from types import SimpleNamespace

import pytest

from lingxi.evals.case import Case
from lingxi.evals.cli import fisher_p, load_baseline, save_baseline
from lingxi.evals.runner import CaseScore, check_controls, score_case


def _case(detect, controls):
    return Case(id="c", symptom="s", persona="config/personas/example_persona.yaml",
                recipient="feishu:oc_x", clock=datetime(2026, 1, 1, 12),
                detect=detect, controls=controls, input="在吗")


@pytest.mark.asyncio
async def test_a_regex_that_misses_its_own_failure_is_caught():
    case = _case({"fail": {"any_of": ["你也养猫吗"]}},
                 {"fires": ["你什么时候养的猫"], "quiet": ["团子好可爱"]})
    assert "该判失败的没判出来" in await check_controls(case)


@pytest.mark.asyncio
async def test_a_detector_that_fires_on_a_good_reply_is_caught():
    case = _case({"fail": {"any_of": ["猫"]}},
                 {"fires": ["你也养猫吗"], "quiet": ["团子是只猫吧 好可爱"]})
    assert "不该判失败的被判了" in await check_controls(case)


@pytest.mark.asyncio
async def test_a_judge_is_taken_by_majority():
    class _Judge:
        def __init__(self):
            self.n = 0

        async def complete(self, **kw):
            self.n += 1
            # wrong once in five on every text
            yes = "你也养猫吗" in kw["messages"][0]["content"]
            return SimpleNamespace(content=("NO" if yes else "YES") if self.n % 5 == 0
                                   else ("YES" if yes else "NO"))

    case = _case({"fail": {"judge": "她是不是忘了对方养猫？"}},
                 {"fires": ["你也养猫吗"], "quiet": ["团子好可爱"], "votes": 5})
    assert await check_controls(case, _Judge()) == ""


@pytest.mark.asyncio
async def test_failed_controls_make_the_case_broken_before_sampling():
    sampled = []

    async def sampler(system, messages, n):
        sampled.append(n)
        return ["x"] * n

    case = _case({"fail": {"any_of": ["你也养猫吗"]}}, {"fires": ["你什么时候养的猫"]})
    score = await score_case(case, sampler=sampler)
    assert score.verdict == "BROKEN" and sampled == []


def test_recording_one_case_keeps_the_others(tmp_path):
    path = tmp_path / "baseline.json"
    save_baseline(path, [CaseScore(id="a", verdict="PASS", fail_rate=0.1, samples=10)], "x")
    save_baseline(path, [CaseScore(id="b", verdict="FAIL", fail_rate=0.5, samples=10)], "y")
    assert set(load_baseline(path)) == {"a", "b"}


def test_fisher_separates_noise_from_change():
    assert fisher_p(2, 12, 3, 12) > 0.5          # 17% vs 25%: noise
    assert fisher_p(0, 20, 10, 20) < 0.01        # 0% vs 50%: real
