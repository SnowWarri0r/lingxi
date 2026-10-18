"""What she remembers about him is settled once the conversation is over.

Per-turn, append-only writing left one user with 39 facts: a trip still
planned weeks after it happened, one story as eight snapshots, and things he
did for a real performer credited to the persona she voices. The consolidator
reads the whole conversation with the facts already held and returns edits.
"""

from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from lingxi.brain.orchestrator import build_orchestrator_prompt, StateDigest
from lingxi.facts.models import FactType, Source
from lingxi.facts.store import FactStore
from lingxi.facts.writers.user_statement import UserStatementWriter
from lingxi.memory.consolidator import (
    BACKLOG_TURNS,
    QUIET_AFTER,
    ConsolidationScheduler,
    HeldFact,
    Turn,
    build_prompt,
    due,
    parse_ops,
    real_people,
)
from lingxi.memory.short_term import ConversationTurn

T0 = datetime(2026, 9, 5, 20, 0)
SUBJ = "user:feishu:oc_x"


def _turns():
    return [
        Turn("assistant", "我今天去填了下个月漫展的抽选", T0),
        Turn("user", "我周六去了漫展，见到阿澪本人了", T0 + timedelta(minutes=1)),
        Turn("user", "还把写好的信给她了", T0 + timedelta(minutes=2)),
    ]


# --- the prompt ------------------------------------------------------------

def test_held_facts_are_shown_with_short_ids_and_dates():
    prompt, ids = build_prompt(
        _turns(), [HeldFact("real-id-1", "对方周六要去漫展", datetime(2026, 9, 1))],
        agent="可可")
    assert "F1. [9月1日记] 对方周六要去漫展" in prompt
    assert ids == {"F1": "real-id-1"}


def test_the_conversation_is_dated_and_each_line_says_who_spoke():
    prompt, _ = build_prompt(_turns(), [], agent="可可")
    assert "2026年9月5日 周六" in prompt
    assert "20:01 对方：我周六去了漫展" in prompt
    assert "20:00 可可：我今天去填了" in prompt


def test_real_people_are_named_as_other_people():
    prompt, _ = build_prompt(_turns(), [], agent="可可",
                             people=["阿澪 / Mio：演你自己的人"])
    assert "跟可可不是同一个人" in prompt
    assert "阿澪 / Mio：演你自己的人" in prompt


def test_no_people_block_when_the_persona_names_none():
    prompt, _ = build_prompt(_turns(), [], agent="可可")
    assert "不是同一个人】" not in prompt


def test_real_people_come_from_the_tagged_lexicon_lines():
    persona = SimpleNamespace(lexicon=["【台下】推し：最喜欢的那一个",
                                       "【中の人】阿澪 / Mio：演你自己的人"])
    assert real_people(persona) == ["阿澪 / Mio：演你自己的人"]


# --- parsing the edits -------------------------------------------------------

HIS = ["我周六去了漫展，见到阿澪本人了", "还把写好的信给她了"]
IDS = {"F1": "id-1", "F2": "id-2"}


def test_the_three_edits_parse():
    ops = parse_ops('''{"ops": [
        {"op": "add", "content": "对方把信交给了阿澪", "evidence": "把写好的信给她了"},
        {"op": "update", "id": "F1", "content": "对方9月5日去了漫展，见到了阿澪", "evidence": "我周六去了漫展"},
        {"op": "retire", "id": "F2"}]}''', IDS, his_words=HIS)
    assert [(o.op, o.target) for o in ops] == [
        ("add", ""), ("update", "id-1"), ("retire", "id-2")]


def test_her_own_news_cannot_be_recorded_as_his():
    """The quote has to be found in what he said, not in what she said."""
    ops = parse_ops('{"ops": [{"op": "add", "content": "对方填了漫展的抽选", '
                    '"evidence": "去填了下个月漫展的抽选"}]}', IDS, his_words=HIS)
    assert ops == []


def test_an_edit_with_no_quote_is_dropped():
    ops = parse_ops('{"ops": [{"op": "update", "id": "F1", '
                    '"content": "对方去了漫展"}]}', IDS, his_words=HIS)
    assert ops == []


def test_a_quote_survives_spacing_and_quotation_marks():
    ops = parse_ops('{"ops": [{"op": "add", "content": "对方见到了阿澪", '
                    '"evidence": "「见到阿澪 本人了」"}]}', IDS, his_words=HIS)
    assert len(ops) == 1


def test_a_retirement_needs_no_quote():
    ops = parse_ops('{"ops": [{"op": "retire", "id": "F2"}]}', IDS, his_words=HIS)
    assert [o.op for o in ops] == ["retire"]


def test_an_edit_to_a_fact_that_was_not_shown_is_dropped():
    ops = parse_ops('{"ops": [{"op": "retire", "id": "F9"}]}', IDS, his_words=HIS)
    assert ops == []


def test_one_fact_is_edited_once():
    ops = parse_ops('{"ops": [{"op": "retire", "id": "F1"}, '
                    '{"op": "update", "id": "F1", "content": "对方去了漫展", '
                    '"evidence": "去了漫展"}]}', IDS, his_words=HIS)
    assert [o.op for o in ops] == ["retire"]


def test_broken_json_edits_nothing():
    assert parse_ops("not json", IDS, his_words=HIS) == []


# --- when it runs ------------------------------------------------------------

def _rec(last_interaction, consolidated=None):
    return SimpleNamespace(last_interaction=last_interaction,
                           last_consolidated_at=consolidated)


def _ct(role, minutes):
    return ConversationTurn(role=role, content="嗯", timestamp=T0 + timedelta(minutes=minutes))


def test_not_while_he_is_still_talking():
    turns = [_ct("user", 0), _ct("assistant", 1)]
    assert due(_rec(T0), turns, T0 + QUIET_AFTER - timedelta(minutes=1)) is None


def test_once_he_has_gone_quiet():
    turns = [_ct("user", 0), _ct("assistant", 1)]
    assert due(_rec(T0), turns, T0 + QUIET_AFTER) == turns


def test_only_the_turns_not_yet_read():
    turns = [_ct("user", 0), _ct("assistant", 1), _ct("user", 60)]
    fresh = due(_rec(turns[-1].timestamp, consolidated=turns[1].timestamp),
                turns, T0 + timedelta(minutes=60) + QUIET_AFTER)
    assert fresh == turns[2:]


def test_her_unanswered_openers_alone_are_not_a_conversation():
    turns = [_ct("user", 0), _ct("assistant", 120), _ct("assistant", 400)]
    rec = _rec(T0, consolidated=turns[0].timestamp)
    assert due(rec, turns, T0 + timedelta(hours=10)) is None


def test_a_long_conversation_is_read_before_the_buffer_drops_its_start():
    turns = [_ct("user" if i % 2 else "assistant", i) for i in range(BACKLOG_TURNS)]
    assert due(_rec(turns[-1].timestamp), turns, turns[-1].timestamp) == turns


# --- applying the edits -------------------------------------------------------

class _LLM:
    def __init__(self, content):
        self.content = content
        self.prompts = []

    async def complete(self, **kw):
        self.prompts.append(kw["messages"][0]["content"])
        return SimpleNamespace(content=self.content)


class _Tracker:
    def __init__(self, rec):
        self.rec = rec
        self.saved = 0

    def all_records(self):
        return [self.rec]

    async def save(self):
        self.saved += 1


class _ShortTerm:
    def __init__(self, turns):
        self.turns = turns

    async def snapshot_for_recipient(self, key):
        return self.turns


@pytest.mark.asyncio
async def test_a_quiet_conversation_rewrites_the_plan_it_carried_out(tmp_path):
    store = FactStore(tmp_path / "facts.db")
    await store.init()
    writer = UserStatementWriter(store)
    plan = await writer.write(subject=SUBJ, content="对方周六要去漫展",
                              type=FactType.PATTERN, source=Source.USER_STATED,
                              ts=T0 - timedelta(days=5))
    dup = await writer.write(subject=SUBJ, content="对方这周六去漫展",
                             type=FactType.PATTERN, source=Source.USER_STATED,
                             ts=T0 - timedelta(days=4))
    llm = _LLM('''{"ops": [
        {"op": "update", "id": "F2", "content": "对方9月5日去了漫展，见到了阿澪", "evidence": "我周六去了漫展"},
        {"op": "retire", "id": "F1"},
        {"op": "add", "content": "对方给阿澪写了信并当面交给了她", "evidence": "把写好的信给她了"}]}''')
    turns = [ConversationTurn(role=t.role, content=t.content, timestamp=t.ts)
             for t in _turns()]
    rec = SimpleNamespace(channel="feishu", recipient_id="oc_x",
                          last_interaction=turns[-1].timestamp,
                          last_consolidated_at=None, unreachable_since=None,
                          last_review_date=turns[-1].timestamp.date().isoformat())
    tracker = _Tracker(rec)
    sched = ConsolidationScheduler(
        llm=llm, tracker=tracker, short_term=_ShortTerm(turns), store=store,
        writer=writer, persona=SimpleNamespace(name="可可", lexicon=[]))

    await sched.tick(turns[-1].timestamp + QUIET_AFTER)

    held = sorted(f.content for f in await store.query(subject=SUBJ, type=FactType.PATTERN))
    assert held == ["对方9月5日去了漫展，见到了阿澪", "对方给阿澪写了信并当面交给了她"]
    # Nothing is deleted: the rewritten and the retired rows are still there.
    every = await store.query(subject=SUBJ, exclude_superseded=False,
                              include_expired=True)
    assert {plan.id, dup.id} <= {f.id for f in every}
    assert rec.last_consolidated_at == turns[-1].timestamp
    assert tracker.saved == 1

    # The same conversation is not read twice.
    await sched.tick(turns[-1].timestamp + QUIET_AFTER * 2)
    assert len(llm.prompts) == 1


@pytest.mark.asyncio
async def test_a_chat_that_can_no_longer_be_reached_is_skipped(tmp_path):
    llm = _LLM('{"ops": []}')
    rec = SimpleNamespace(channel="feishu", recipient_id="oc_x",
                          last_interaction=T0, last_consolidated_at=None,
                          unreachable_since=T0)
    sched = ConsolidationScheduler(
        llm=llm, tracker=_Tracker(rec), short_term=_ShortTerm([_ct("user", 0)]),
        store=None, writer=None, persona=SimpleNamespace(name="可可", lexicon=[]))
    await sched.tick(T0 + timedelta(hours=1))
    assert llm.prompts == []


# --- the per-turn path no longer writes --------------------------------------

def test_the_orchestrator_is_not_asked_to_write_memories():
    prompt = build_orchestrator_prompt(
        "我周六去了漫展", StateDigest(activity="", mood="", last_lived=[]), {})
    assert "memory_writes" not in prompt


# --- the nightly review --------------------------------------------------------

from lingxi.memory.consolidator import (  # noqa: E402
    REVIEW_HOUR, adds_only_time, parse_review, review_due, review_prompt)


def test_a_past_plan_may_become_not_known_how_it_went():
    assert adds_only_time("对方说国庆会去漫展见阿澪",
                          "对方说过国庆会去漫展见阿澪，后来去没去还不知道")


def test_a_review_cannot_invent_how_it_went():
    assert not adds_only_time("对方说国庆会去漫展见阿澪",
                              "对方国庆去了漫展，见到了阿澪本人")


def test_a_relative_day_may_become_a_date():
    assert adds_only_time("对方昨天剪了头发", "对方8月15日剪了头发")


def test_review_edits_parse_and_an_invented_outcome_is_dropped():
    ids = {"F1": "a", "F2": "b", "F3": "c"}
    held = {"a": "对方说国庆会去漫展", "b": "对方这几天感冒了", "c": "对方住在邻市"}
    ops = parse_review('''{"ops": [
        {"op": "update", "id": "F1", "content": "对方国庆去了漫展，玩得很开心"},
        {"op": "retire", "id": "F2"},
        {"op": "add", "content": "对方喜欢猫"}]}''', ids, held)
    assert [(o.op, o.target) for o in ops] == [("retire", "b")]


def test_the_review_is_told_the_date_and_when_each_fact_was_written():
    prompt, ids = review_prompt(
        [HeldFact("a", "对方昨天剪了头发", datetime(2026, 8, 16))],
        agent="可可", today=datetime(2026, 10, 14, 4))
    assert "2026年10月14日" in prompt and "[2026年8月16日记]" in prompt


def test_once_a_night_after_the_review_hour():
    rec = SimpleNamespace(last_review_date=None)
    night = datetime(2026, 10, 14, REVIEW_HOUR)
    assert not review_due(rec, night - timedelta(hours=1))
    assert review_due(rec, night)
    rec.last_review_date = "2026-10-14"
    assert not review_due(rec, night + timedelta(hours=10))
    assert review_due(rec, night + timedelta(days=1))


@pytest.mark.asyncio
async def test_the_nightly_pass_retires_a_passing_state(tmp_path):
    store = FactStore(tmp_path / "facts.db")
    await store.init()
    writer = UserStatementWriter(store)
    cold = await writer.write(subject=SUBJ, content="对方这几天感冒了",
                              type=FactType.PATTERN, source=Source.USER_STATED,
                              ts=T0)
    rec = SimpleNamespace(channel="feishu", recipient_id="oc_x",
                          last_interaction=T0, last_consolidated_at=T0,
                          last_review_date=None, unreachable_since=None)
    tracker = _Tracker(rec)
    sched = ConsolidationScheduler(
        llm=_LLM('{"ops": [{"op": "retire", "id": "F1"}]}'), tracker=tracker,
        short_term=_ShortTerm([]), store=store, writer=writer,
        persona=SimpleNamespace(name="可可", lexicon=[]))
    night = datetime(2026, 10, 14, REVIEW_HOUR)
    await sched.tick(night)
    assert (await store.get(cold.id)).expires_at == night
    assert rec.last_review_date == "2026-10-14"
    await sched.tick(night + timedelta(hours=1))
    assert len(sched._llm.prompts) == 1
