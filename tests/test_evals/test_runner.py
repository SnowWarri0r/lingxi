from datetime import timedelta

import pytest

from lingxi.evals.case import load_case
from lingxi.evals.runner import build_turn, score_case

YAML = """
id: t
symptom: s
persona: config/personas/tangkeke.yaml
recipient: feishu:oc_eval
clock: "2026-08-19T20:20:54"
facts:
  - {subject: "user:feishu:oc_eval", type: pattern, source: user_stated,
     content: 对方一般晚上九点下班, importance: 4, days_ago: 12}
history:
  - {role: user, content: 想下班了, minutes_ago: 2}
input: 大学还不轻松啊
samples: 4
premise:
  prompt_contains: ["2026-08-19 20:20"]
detect:
  fail: {any_of: [堵车]}
  pass: {any_of: [还在公司]}
budget: {max_fail_rate: 0.25}
"""


def _case(tmp_path, text=YAML):
    p = tmp_path / "c.yaml"
    p.write_text(text, encoding="utf-8")
    return load_case(p)


def _sampler(replies):
    async def _s(system, messages, n):
        return (replies * n)[:n]
    return _s


class _StubLLM:
    """The orchestrator is monkeypatched in these tests, so nothing calls it.

    Injected anyway: the real _main_llm() resolves OAuth credentials, which
    an offline test must not depend on.
    """

    async def complete(self, **kwargs):
        raise AssertionError("orchestrator should be monkeypatched in tests")


@pytest.fixture(autouse=True)
def _clean_weather_cache():
    """Isolate every test from the module-global weather cache.

    build_turn runs the real prompt pipeline, which reads
    lingxi.temporal.weather's in-process cache for a "current weather" line.
    That cache is shared process-wide (keyed by rounded lat/lon), so without
    this fixture a test's outcome would depend on whatever another test file
    happened to leave behind — an offline harness must not have its
    determinism riding on pytest's collection order. Clearing before AND
    after also protects test_temporal/test_weather.py from any residue this
    file leaves via the explicit seeding test below.
    """
    from lingxi.temporal import weather
    weather._cache.clear()
    yield
    weather._cache.clear()


@pytest.mark.asyncio
async def test_build_turn_puts_case_facts_in_reach(tmp_path, monkeypatch):
    """The case's own facts must be the ones assembled — not the live db."""
    from lingxi.brain import orchestrator as orch_mod
    from lingxi.brain.models import OrchestrationDecision

    async def _fake_decide(*a, **k):
        return OrchestrationDecision(
            register="warm", engage_level=0.6, fact_queries=[], skip=[],
            topic_anchor="")

    monkeypatch.setattr(orch_mod, "decide", _fake_decide)
    system, messages, persona = await build_turn(_case(tmp_path), llm=_StubLLM())
    assert "唐可可" in system or persona.name
    assert "九点下班" in messages[-1]["content"]


@pytest.mark.asyncio
async def test_two_builds_are_identical(tmp_path, monkeypatch):
    from lingxi.brain import orchestrator as orch_mod
    from lingxi.brain.models import OrchestrationDecision

    async def _fake_decide(*a, **k):
        return OrchestrationDecision(
            register="warm", engage_level=0.6, fact_queries=[], skip=[],
            topic_anchor="")

    monkeypatch.setattr(orch_mod, "decide", _fake_decide)
    a_sys, a_msgs, _ = await build_turn(_case(tmp_path), llm=_StubLLM())
    b_sys, b_msgs, _ = await build_turn(_case(tmp_path), llm=_StubLLM())
    assert a_sys == b_sys
    assert a_msgs[-1]["content"] == b_msgs[-1]["content"]


@pytest.mark.asyncio
async def test_pass_verdict_when_under_budget(tmp_path, monkeypatch):
    from lingxi.brain import orchestrator as orch_mod
    from lingxi.brain.models import OrchestrationDecision

    async def _fake_decide(*a, **k):
        return OrchestrationDecision(
            register="warm", engage_level=0.6, fact_queries=[], skip=[],
            topic_anchor="")

    monkeypatch.setattr(orch_mod, "decide", _fake_decide)
    score = await score_case(
        _case(tmp_path), sampler=_sampler(["还在公司蹲着"]), llm=_StubLLM())
    assert score.verdict == "PASS"
    assert score.fail_rate == 0.0
    assert score.pass_rate == 1.0


@pytest.mark.asyncio
async def test_fail_verdict_when_over_budget(tmp_path, monkeypatch):
    from lingxi.brain import orchestrator as orch_mod
    from lingxi.brain.models import OrchestrationDecision

    async def _fake_decide(*a, **k):
        return OrchestrationDecision(
            register="warm", engage_level=0.6, fact_queries=[], skip=[],
            topic_anchor="")

    monkeypatch.setattr(orch_mod, "decide", _fake_decide)
    score = await score_case(_case(tmp_path), sampler=_sampler(["还在堵车"]),
                             llm=_StubLLM())
    assert score.verdict == "FAIL"
    assert score.fail_rate == 1.0


@pytest.mark.asyncio
async def test_broken_when_premise_fails(tmp_path, monkeypatch):
    """A case whose premise no longer holds reports BROKEN, never PASS."""
    from lingxi.brain import orchestrator as orch_mod
    from lingxi.brain.models import OrchestrationDecision

    async def _fake_decide(*a, **k):
        return OrchestrationDecision(
            register="warm", engage_level=0.6, fact_queries=[], skip=[],
            topic_anchor="")

    monkeypatch.setattr(orch_mod, "decide", _fake_decide)
    bad = YAML.replace('["2026-08-19 20:20"]', '["这句话不可能出现在 prompt 里"]')
    score = await score_case(_case(tmp_path, bad), sampler=_sampler(["随便"]),
                             llm=_StubLLM())
    assert score.verdict == "BROKEN"
    assert score.premise_ok is False
    assert "这句话不可能出现在 prompt 里" in score.premise_error


@pytest.mark.asyncio
async def test_pass_rate_does_not_affect_verdict(tmp_path, monkeypatch):
    """pass is observation only: 0 pass hits with 0 fail hits is still PASS."""
    from lingxi.brain import orchestrator as orch_mod
    from lingxi.brain.models import OrchestrationDecision

    async def _fake_decide(*a, **k):
        return OrchestrationDecision(
            register="warm", engage_level=0.6, fact_queries=[], skip=[],
            topic_anchor="")

    monkeypatch.setattr(orch_mod, "decide", _fake_decide)
    score = await score_case(_case(tmp_path), sampler=_sampler(["今天好热"]),
                             llm=_StubLLM())
    assert score.verdict == "PASS"
    assert score.pass_rate == 0.0


@pytest.mark.asyncio
async def test_build_turn_stubs_out_weather(tmp_path, monkeypatch):
    """Spec §5.5: weather must be stubbed out so the same case assembles the
    same prompt whether or not the live lingxi-feishu bot happens to be
    running on this box and has populated lingxi.temporal.weather's
    process-global cache.

    Seed the cache with a recognisable reading for the case's persona
    location — the same setup that would make the live pipeline surface a
    weather line — and confirm build_turn's assembled prompt does NOT carry
    it. This must fail if the `_weather_line` override in build_turn is
    removed.
    """
    from lingxi.brain import orchestrator as orch_mod
    from lingxi.brain.models import OrchestrationDecision
    from lingxi.persona.loader import load_persona
    from lingxi.temporal import weather
    from lingxi.temporal.sun import persona_location

    async def _fake_decide(*a, **k):
        return OrchestrationDecision(
            register="warm", engage_level=0.6, fact_queries=[], skip=[],
            topic_anchor="")

    monkeypatch.setattr(orch_mod, "decide", _fake_decide)

    case = _case(tmp_path)
    loc = persona_location(load_persona(case.persona))
    stub = weather.Weather(
        temp_c=31.0, feels_like_c=31.0, description="大雨",
        wind_kmh=5.0, is_day=True, fetched_at=case.clock - timedelta(minutes=5),
    )
    weather._cache[weather._key(loc)] = stub

    system, messages, _ = await build_turn(case, llm=_StubLLM())
    full = system + "\n" + messages[-1]["content"]
    assert "大雨" not in full
    assert "31°C" not in full


@pytest.mark.asyncio
async def test_build_turn_keeps_sunrise_sunset_live(tmp_path, monkeypatch):
    """Spec §5.5: unlike weather, sunrise/sunset must stay live — it is pure
    offline computation from the frozen clock and the persona's location, so
    it costs nothing to keep deterministic and is worth exercising end to
    end. A future change that stubs "all of temporal" wholesale must not be
    able to silently take this out along with weather.
    """
    from lingxi.brain import orchestrator as orch_mod
    from lingxi.brain.models import OrchestrationDecision
    from lingxi.persona.loader import load_persona
    from lingxi.persona.prompt_builder import PromptBuilder

    async def _fake_decide(*a, **k):
        return OrchestrationDecision(
            register="warm", engage_level=0.6, fact_queries=[], skip=[],
            topic_anchor="")

    monkeypatch.setattr(orch_mod, "decide", _fake_decide)

    case = _case(tmp_path)
    persona = load_persona(case.persona)
    expected_scene = PromptBuilder(persona)._daylight_scene(case.clock)

    system, messages, _ = await build_turn(case, llm=_StubLLM())
    full = system + "\n" + messages[-1]["content"]
    assert expected_scene in full
