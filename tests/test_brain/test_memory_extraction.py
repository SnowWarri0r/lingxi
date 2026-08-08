"""Durable user facts are extracted by the orchestrator.

Regression: `memory_writes` existed in the responder's META schema with no
instruction anywhere explaining it, so it was never filled — after months of
chat, facts.db held 376 of the persona's own events and zero facts about the
user. Extraction now rides the orchestrator's pre-turn analysis.
"""
import inspect

from lingxi.brain import orchestrator
from lingxi.brain.models import OrchestrationDecision
from lingxi.conversation.engine import ConversationEngine


def test_decision_parses_and_cleans_memory_writes():
    d = OrchestrationDecision.from_dict({
        "engage_level": 0.6, "register": "warm", "fact_queries": [],
        "topic_anchor": "", "skip": [],
        "memory_writes": ["  对方一般七点半下班  ", "", "   ", "对方养了只猫"],
    })
    assert d.memory_writes == ["对方一般七点半下班", "对方养了只猫"]


def test_memory_writes_defaults_empty():
    assert OrchestrationDecision.default().memory_writes == []
    d = OrchestrationDecision.from_dict({
        "engage_level": 0.6, "register": "warm", "fact_queries": [],
        "topic_anchor": "", "skip": [],
    })
    assert d.memory_writes == []


def test_orchestrator_prompt_instructs_extraction():
    p = orchestrator._PROMPT
    assert "memory_writes" in p
    # the field must be explained, not just present in the JSON skeleton
    assert p.count("memory_writes") >= 2
    assert "下次还用得上" in p
    # no inventing unstated attributes about the user
    assert "只写用户自己说过的" in p


def test_engine_writes_orchestrator_facts():
    src = inspect.getsource(ConversationEngine)
    assert "self._write_user_facts(decision.memory_writes)" in src
    # and a failed write is reported rather than dying inside the task
    assert "user-fact write failed" in src
