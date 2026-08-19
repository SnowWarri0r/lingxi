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


def test_light_register_is_accepted():
    # "light" is the casual-banter register — must survive (not coerce to warm),
    # so everyday chatter doesn't get routed to the deep "warm" 接法.
    raw = {"engage_level": 0.4, "register": "light",
           "fact_queries": [], "topic_anchor": "", "skip": []}
    assert OrchestrationDecision.from_dict(raw).register == "light"


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


def test_lookup_query_parsed_and_defaults_empty():
    # default + no field → empty
    assert OrchestrationDecision.default().lookup_query == ""
    d = OrchestrationDecision.from_dict({
        "engage_level": 0.5, "register": "warm", "fact_queries": [],
        "topic_anchor": "", "skip": [],
    })
    assert d.lookup_query == ""
    # present → parsed & stripped
    d2 = OrchestrationDecision.from_dict({
        "engage_level": 0.6, "register": "curious", "fact_queries": [],
        "topic_anchor": "", "skip": [],
        "lookup_query": "  Love Live Superstar 第一季 东京预选 结果  ",
    })
    assert d2.lookup_query == "Love Live Superstar 第一季 东京预选 结果"
    # null → empty, not the string "None"
    d3 = OrchestrationDecision.from_dict({
        "engage_level": 0.6, "register": "warm", "fact_queries": [],
        "topic_anchor": "", "skip": [], "lookup_query": None,
    })
    assert d3.lookup_query == ""


def _decision(**extra):
    base = {"engage_level": 0.6, "register": "warm", "fact_queries": [],
            "topic_anchor": "", "skip": []}
    base.update(extra)
    return OrchestrationDecision.from_dict(base)


def test_user_state_kept_when_it_commits_to_one_situation():
    assert _decision(user_state="还在公司，想下班还没到点").user_state == \
        "还在公司，想下班还没到点"
    assert _decision(user_state="  在回家的地铁上  ").user_state == "在回家的地铁上"
    assert _decision().user_state == ""
    assert _decision(user_state=None).user_state == ""


def test_user_state_dropped_when_it_hedges():
    """A hedged state reads worse than none — the responder answers the later
    branch. Measured on the failing turn: decisive 1/20 wrong, hedged 5/20,
    no state at all 3/20. So a hedge falls back to the clock instead."""
    for hedged in ("还在公司或刚下班途中",
                   "可能已经到家了",
                   "大概在路上",
                   "到家了？"):
        assert _decision(user_state=hedged).user_state == "", hedged
