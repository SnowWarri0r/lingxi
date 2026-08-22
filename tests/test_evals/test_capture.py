import json
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


def _write_short_term(
    tmp_path: Path, recipient_key: str,
    rows: list[tuple[str, str, datetime]],
) -> None:
    """Write a short_term buffer file in the real on-disk shape:
    {"recipient": ..., "turns": [{"role", "content", "timestamp", ...}, ...]}
    """
    path = tmp_path / "short_term" / f"{recipient_key}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "recipient": recipient_key,
        "turns": [
            {"role": role, "content": content,
             "timestamp": ts.isoformat(), "metadata": {}, "summary": None}
            for role, content, ts in rows
        ],
    }, ensure_ascii=False), encoding="utf-8")


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


@pytest.mark.asyncio
async def test_capture_input_is_the_users_message_not_her_own_reply(tmp_path):
    """The short_term buffer at rest always ends with the assistant's own
    reply (appended right after generation). `input` must be the user's
    last message, not that trailing reply — otherwise a case replays the
    model's own prior line as if the user had just said it."""
    await _live(tmp_path)
    _write_short_term(tmp_path, "feishu:oc_x", [
        ("user", "早上好", AT - timedelta(minutes=30)),
        ("assistant", "早呀", AT - timedelta(minutes=29)),
        ("user", "比昨天要严重一点，头有点昏沉", AT - timedelta(minutes=5)),
        ("assistant", "啊 更严重了吗 头昏沉沉的最难受了", AT - timedelta(minutes=4)),
    ])
    skeleton = await capture(
        "feishu:oc_x", persona_path="config/personas/tangkeke.yaml",
        data_dir=tmp_path, turns=8, at=AT)

    assert skeleton["input"] == "比昨天要严重一点，头有点昏沉"
    # The reply that followed the last user turn must not leak into history.
    contents = [row["content"] for row in skeleton["history"]]
    assert "啊 更严重了吗 头昏沉沉的最难受了" not in contents
    assert contents == ["早上好", "早呀"]


@pytest.mark.asyncio
async def test_capture_no_user_turn_yields_empty_input_and_history(tmp_path):
    """A buffer with no user turn at all (e.g. only proactive messages)
    must not guess at an input; blank is correct, not a crash or a
    fabricated value."""
    await _live(tmp_path)
    _write_short_term(tmp_path, "feishu:oc_x", [
        ("assistant", "早呀，刷牙刷到一半突然哼起歌来了", AT - timedelta(minutes=30)),
        ("assistant", "你今天有空吗", AT - timedelta(minutes=20)),
    ])
    skeleton = await capture(
        "feishu:oc_x", persona_path="config/personas/tangkeke.yaml",
        data_dir=tmp_path, turns=8, at=AT)

    assert skeleton["input"] == ""
    assert skeleton["history"] == []


@pytest.mark.asyncio
async def test_capture_against_missing_facts_db_returns_empty_facts(tmp_path):
    """capture() must initialise the FactStore itself. Against a data_dir
    whose facts.db does not exist yet (a persona that has never written a
    fact), it must return an empty-facts skeleton instead of raising
    sqlite3.OperationalError: no such table: facts."""
    assert not (tmp_path / "facts.db").exists()
    skeleton = await capture(
        "feishu:oc_x", persona_path="config/personas/tangkeke.yaml",
        data_dir=tmp_path, turns=8, at=AT)
    assert skeleton["facts"] == []
    skeleton["symptom"] = "填写：错在哪"
    # A hand-written placeholder standing in for the human's real detect —
    # the empty any_of the raw skeleton ships with is rejected at load time
    # (see test_case_loader.py) and is irrelevant to what this test checks.
    skeleton["detect"] = {"fail": {"any_of": ["占位"]}}
    case = Case.model_validate(yaml.safe_load(yaml.safe_dump(skeleton)))
    assert case.facts == []


@pytest.mark.asyncio
async def test_capture_keeps_minute_precision_for_recent_facts(tmp_path):
    """A fact under a day old must round-trip to within a minute of its
    original timestamp. days_ago rounded to 2 decimals has ~14 minutes of
    slop, which would blur an hour-old fact into meaninglessness."""
    store = FactStore(tmp_path / "facts.db")
    await store.init()
    fact_ts = AT - timedelta(hours=1)
    await store.write(Fact(
        subject="aria", content="刚练完舞蹈回到家", source=Source.LIFE_SIMULATED,
        type=FactType.EVENT, ts=fact_ts, importance=3))

    skeleton = await capture(
        "feishu:oc_x", persona_path="config/personas/tangkeke.yaml",
        data_dir=tmp_path, turns=8, at=AT)
    skeleton["symptom"] = "填写：错在哪"
    # Placeholder detect — see the identical note above; this test is about
    # fact-age round-tripping, not detect validation.
    skeleton["detect"] = {"fail": {"any_of": ["占位"]}}
    case = Case.model_validate(yaml.safe_load(yaml.safe_dump(skeleton)))

    recent_fact = next(f for f in case.facts if "刚练完舞蹈" in f.content)
    assert recent_fact.days_ago == 0.0
    assert recent_fact.minutes_ago == pytest.approx(60.0, abs=1)

    resolved = case.resolved_facts()
    resolved_fact = next(f for f in resolved if "刚练完舞蹈" in f.content)
    delta = abs((resolved_fact.ts - fact_ts).total_seconds())
    assert delta < 60


@pytest.mark.asyncio
async def test_capture_default_fact_limit_is_eight(tmp_path):
    """The default must be small enough that a human can scan the whole
    skeleton, not the old 30-per-subject that buried the two facts that
    matter in fifty narrative ones."""
    store = FactStore(tmp_path / "facts.db")
    await store.init()
    for i in range(15):
        await store.write(Fact(
            subject="aria", content=f"事件 {i}", source=Source.LIFE_SIMULATED,
            type=FactType.EVENT, ts=AT - timedelta(hours=i), importance=1))

    skeleton = await capture(
        "feishu:oc_x", persona_path="config/personas/tangkeke.yaml",
        data_dir=tmp_path, turns=8, at=AT)
    aria_facts = [f for f in skeleton["facts"] if f["subject"] == "aria"]
    assert len(aria_facts) == 8
