from datetime import datetime, timedelta
from pathlib import Path

import pytest
import yaml

from lingxi.evals.capture import capture
from lingxi.evals.case import Case
from lingxi.facts.models import Fact, FactType, Source
from lingxi.facts.store import FactStore

AT = datetime(2026, 8, 19, 20, 20, 54)


async def _live(tmp_path: Path):
    store = FactStore(tmp_path / "facts.db")
    await store.init()
    await store.write(Fact(
        subject="user:feishu:oc_x", content="对方一般晚上九点下班",
        source=Source.USER_STATED, type=FactType.PATTERN,
        ts=AT - timedelta(days=12), importance=4))
    await store.write(Fact(
        subject="aria", content="在练习室抠队形", source=Source.LIFE_SIMULATED,
        type=FactType.EVENT, ts=AT - timedelta(hours=1), importance=4))
    return store


@pytest.mark.asyncio
async def test_capture_produces_a_loadable_skeleton(tmp_path):
    await _live(tmp_path)
    skeleton = await capture(
        "feishu:oc_x", persona_path="config/personas/tangkeke.yaml",
        data_dir=tmp_path, turns=8, at=AT)
    # Author fills these two in; everything else is mechanical.
    skeleton["symptom"] = "填写：错在哪"
    skeleton["detect"] = {"fail": {"any_of": ["堵车"]}}
    case = Case.model_validate(yaml.safe_load(yaml.safe_dump(skeleton)))
    assert case.clock == AT
    assert case.recipient == "feishu:oc_x"


@pytest.mark.asyncio
async def test_capture_converts_fact_ages_to_days_ago(tmp_path):
    await _live(tmp_path)
    skeleton = await capture(
        "feishu:oc_x", persona_path="config/personas/tangkeke.yaml",
        data_dir=tmp_path, turns=8, at=AT)
    user_fact = next(f for f in skeleton["facts"]
                     if "九点下班" in f["content"])
    assert round(user_fact["days_ago"]) == 12


@pytest.mark.asyncio
async def test_capture_includes_both_her_facts_and_his(tmp_path):
    await _live(tmp_path)
    skeleton = await capture(
        "feishu:oc_x", persona_path="config/personas/tangkeke.yaml",
        data_dir=tmp_path, turns=8, at=AT)
    subjects = {f["subject"] for f in skeleton["facts"]}
    assert "aria" in subjects
    assert "user:feishu:oc_x" in subjects


@pytest.mark.asyncio
async def test_capture_leaves_authoring_fields_empty(tmp_path):
    """symptom and detect are the human's job; a filled-in guess would be
    worse than a blank, because it looks done."""
    await _live(tmp_path)
    skeleton = await capture(
        "feishu:oc_x", persona_path="config/personas/tangkeke.yaml",
        data_dir=tmp_path, turns=8, at=AT)
    assert skeleton["symptom"] == ""
    assert skeleton["detect"] == {"fail": {"any_of": []}}
