import pytest

from lingxi.brain.models import OrchestrationDecision, OrchestratorFactQuery


def test_default_decision_is_warm_mid_engagement():
    d = OrchestrationDecision.default()
    assert d.register == "warm"
    assert 0.5 <= d.engage_level <= 0.7
    assert len(d.fact_queries) >= 1  # must surface SOMETHING


def test_decision_from_json_basic():
    raw = {
        "engage_level": 0.7,
        "register": "warm",
        "fact_queries": [
            {"category": "aria.event", "limit": 3}
        ],
        "topic_anchor": "聊到了工作时间",
        "skip": ["world.event"],
    }
    d = OrchestrationDecision.from_dict(raw)
    assert d.engage_level == 0.7
    assert d.register == "warm"
    assert len(d.fact_queries) == 1
    assert d.fact_queries[0].category == "aria.event"
    assert "world.event" in d.skip


def test_decision_handles_unknown_register_gracefully():
    raw = {
        "engage_level": 0.5, "register": "weirdo",
        "fact_queries": [], "topic_anchor": "", "skip": [],
    }
    d = OrchestrationDecision.from_dict(raw)
    assert d.register == "warm"  # fallback


def test_decision_clamps_engage_level():
    raw = {
        "engage_level": 1.5, "register": "warm",
        "fact_queries": [], "topic_anchor": "", "skip": [],
    }
    d = OrchestrationDecision.from_dict(raw)
    assert d.engage_level == 1.0


def test_decision_has_plan_conflict_default_false():
    from lingxi.brain.models import OrchestrationDecision
    d = OrchestrationDecision.default()
    assert d.plan_conflict is False


def test_decision_from_dict_parses_plan_conflict():
    from lingxi.brain.models import OrchestrationDecision
    d = OrchestrationDecision.from_dict({"plan_conflict": True})
    assert d.plan_conflict is True


def test_decision_from_dict_defaults_plan_conflict_false():
    from lingxi.brain.models import OrchestrationDecision
    d = OrchestrationDecision.from_dict({})
    assert d.plan_conflict is False
