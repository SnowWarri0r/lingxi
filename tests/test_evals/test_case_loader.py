from datetime import datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from lingxi.evals.case import Case, load_all_cases, load_case

YAML = """
id: offwork-state
symptom: 他说想下班 90 秒后，她问是在堵车还是到家了
origin: 2026-08-19 飞书对话
persona: config/personas/tangkeke.yaml
recipient: feishu:oc_eval
clock: "2026-08-19T20:20:54"
facts:
  - subject: "user:feishu:oc_eval"
    type: pattern
    source: user_stated
    content: 对方一般晚上九点下班
    importance: 4
    days_ago: 12
history:
  - {role: user, content: 想下班了, minutes_ago: 2}
  - {role: assistant, content: 想下班的心我懂, minutes_ago: 2}
input: 大学还不轻松啊
samples: 20
premise:
  prompt_contains: ["下班后的个人时间"]
  prompt_lacks: ["还在公司"]
detect:
  fail: {any_of: [堵车, 到家]}
  pass: {any_of: [快下班]}
budget: {max_fail_rate: 0.05}
"""


def _write(tmp_path: Path, text: str, name: str = "c.yaml") -> Path:
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def test_loads_all_fields(tmp_path):
    case = load_case(_write(tmp_path, YAML))
    assert case.id == "offwork-state"
    assert case.clock == datetime(2026, 8, 19, 20, 20, 54)
    assert case.samples == 20
    assert case.budget.max_fail_rate == 0.05
    assert case.premise.prompt_contains == ["下班后的个人时间"]


def test_relative_fact_time_resolves_against_the_frozen_clock(tmp_path):
    case = load_case(_write(tmp_path, YAML))
    fact = case.resolved_facts()[0]
    assert fact.ts == datetime(2026, 8, 7, 20, 20, 54)
    assert fact.subject == "user:feishu:oc_eval"
    assert fact.importance == 4


def test_resolved_facts_never_expire(tmp_path):
    """A case left on disk for a month must not lose facts to the expiry
    filter — that would silently change what the case tests."""
    case = load_case(_write(tmp_path, YAML))
    assert all(f.expires_at is None for f in case.resolved_facts())


def test_relative_history_time_resolves(tmp_path):
    case = load_case(_write(tmp_path, YAML))
    role, content, ts = case.resolved_history()[0]
    assert (role, content) == ("user", "想下班了")
    assert ts == datetime(2026, 8, 19, 20, 18, 54)


def test_samples_defaults_to_20(tmp_path):
    case = load_case(_write(tmp_path, YAML.replace("samples: 20\n", "")))
    assert case.samples == 20


def test_missing_required_field_raises(tmp_path):
    with pytest.raises(ValidationError):
        load_case(_write(tmp_path, YAML.replace("id: offwork-state\n", "")))


def test_unknown_field_raises(tmp_path):
    """Typos must fail loudly — a silently ignored key is a case that tests
    something other than what its author wrote."""
    with pytest.raises(ValidationError):
        load_case(_write(tmp_path, YAML + "\nsampels: 5\n"))


def test_load_all_reads_a_directory(tmp_path):
    _write(tmp_path, YAML, "a.yaml")
    _write(tmp_path, YAML.replace("offwork-state", "second"), "b.yaml")
    ids = sorted(c.id for c in load_all_cases(tmp_path))
    assert ids == ["offwork-state", "second"]


def test_duplicate_ids_raise(tmp_path):
    _write(tmp_path, YAML, "a.yaml")
    _write(tmp_path, YAML, "b.yaml")
    with pytest.raises(ValueError, match="duplicate case id"):
        load_all_cases(tmp_path)
