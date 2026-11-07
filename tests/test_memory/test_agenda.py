"""His plans and her promises are followed up on their day, once, and left
alone when he does not answer."""

import json
from datetime import date, datetime, timedelta
from types import SimpleNamespace

import pytest

from lingxi.memory import agenda as ag
from lingxi.memory.agenda import HERS, HIS, AgendaBook
from lingxi.memory.consolidator import Turn

KEY = "feishu:oc_x"
SAID = datetime(2026, 3, 2, 15, 0)
SHOW = "2026-03-14"


def _book(tmp_path, *items):
    book = AgendaBook(tmp_path / "agenda.json")
    for kind, text, day in items:
        book.add(KEY, kind, text, day, "原话", SAID)
    book.save()
    return book


def _never(at):
    return None


# --- what is due ------------------------------------------------------------

@pytest.mark.parametrize("today,expected", [
    ("2026-03-12", None), ("2026-03-13", "before"), ("2026-03-14", "day"),
    ("2026-03-15", "after"), ("2026-03-17", "after"), ("2026-03-18", None)])
def test_his_plan_is_due_the_day_before_on_the_day_and_after(today, expected):
    item = ag.Item("i", HIS, "对方去漫展见阿澪", SHOW, "", "")
    assert ag.stage(item, date.fromisoformat(today)) == expected


def test_her_promise_is_due_on_its_day_only():
    item = ag.Item("i", HERS, "替他排队买团子周边", SHOW, "", "")
    assert ag.stage(item, date(2026, 3, 13)) is None
    assert ag.stage(item, date(2026, 3, 14)) == "day"


def test_one_item_a_day_and_each_stage_once(tmp_path):
    book = _book(tmp_path, (HIS, "对方去漫展见阿澪", SHOW))
    eve = datetime(2026, 3, 13, 11)
    [item] = book.open_items(KEY, eve.date())
    assert ag.due_today([item], eve, _never) is item
    book.mark_asked(KEY, item.id, "before", eve)
    assert ag.due_today([item], eve + timedelta(hours=5), _never) is None
    assert ag.due_today([item], eve + timedelta(days=1), lambda at: True) is item


def test_a_follow_up_he_did_not_answer_is_not_chased(tmp_path):
    book = _book(tmp_path, (HIS, "对方去漫展见阿澪", SHOW))
    eve = datetime(2026, 3, 13, 11)
    [item] = book.open_items(KEY, eve.date())
    book.mark_asked(KEY, item.id, "before", eve)
    assert ag.due_today([item], eve + timedelta(days=1), lambda at: False) is None


def test_a_plan_long_past_lapses(tmp_path):
    book = _book(tmp_path, (HIS, "对方去漫展见阿澪", SHOW))
    assert book.open_items(KEY, date(2026, 3, 25)) == []
    again = AgendaBook(tmp_path / "agenda.json")
    assert again.items(KEY)[0].status == "lapsed"


# --- the pass after a conversation -----------------------------------------

HE = ["我14号也要去漫展见阿澪", "去了 人好多"]
SHE = ["那我替你排队买团子周边！"]


def _ids(book, today=date(2026, 3, 2)):
    return {f"A{n + 1}": i.id for n, i in enumerate(book.open_items(KEY, today))}


def test_an_add_must_quote_its_speaker():
    text = json.dumps({"add": [
        {"kind": "his", "text": "对方3月14日去漫展见阿澪", "day": SHOW, "evidence": "14号也要去漫展"},
        {"kind": "his", "text": "对方3月14日买团子周边", "day": SHOW, "evidence": "替你排队买团子周边"},
        {"kind": "hers", "text": "替他排队买团子周边", "day": SHOW, "evidence": "替你排队买团子周边"},
    ]}, ensure_ascii=False)
    e = ag.parse_edits(text, {}, his=HE, hers=SHE, said_on=date(2026, 3, 2))
    assert [(a["kind"], a["text"]) for a in e.add] == [
        ("his", "对方3月14日去漫展见阿澪"), ("hers", "替他排队买团子周边")]


def test_a_day_already_behind_the_conversation_is_not_added():
    text = json.dumps({"add": [{"kind": "his", "text": "对方去了漫展", "day": "2026-03-01",
                                "evidence": "14号也要去漫展"}]}, ensure_ascii=False)
    assert ag.parse_edits(text, {}, his=HE, hers=SHE, said_on=date(2026, 3, 2)).add == []


def test_his_plan_is_closed_only_by_his_own_words(tmp_path):
    book = _book(tmp_path, (HIS, "对方去漫展见阿澪", SHOW))
    ids = _ids(book)
    kinds = {i.id: i.kind for i in book.items(KEY)}
    hers = json.dumps({"close": [{"id": "A1", "outcome": "去了", "evidence": "替你排队"}]},
                      ensure_ascii=False)
    his = json.dumps({"close": [{"id": "a1", "outcome": "去了，人很多", "evidence": "人好多"}]},
                     ensure_ascii=False)
    assert ag.parse_edits(hers, ids, his=HE, hers=SHE, said_on=date(2026, 3, 15),
                          kinds=kinds).close == []
    assert ag.parse_edits(his, ids, his=HE, hers=SHE, said_on=date(2026, 3, 15),
                          kinds=kinds).close == [(ids["A1"], "去了，人很多")]


class _LLM:
    def __init__(self, answer):
        self.answer, self.prompts = answer, []

    async def complete(self, **kw):
        self.prompts.append(kw["messages"][0]["content"])
        return SimpleNamespace(content=self.answer)


@pytest.mark.asyncio
async def test_update_writes_the_book_and_shows_what_is_open(tmp_path):
    book = _book(tmp_path, (HIS, "对方去漫展见阿澪", SHOW))
    llm = _LLM(json.dumps({"close": [{"id": "A1", "outcome": "去了", "evidence": "人好多"}]},
                          ensure_ascii=False))
    turns = [Turn("user", "去了 人好多", datetime(2026, 3, 15, 20))]
    await ag.update(llm, book, KEY, turns, agent="可可", now=datetime(2026, 3, 15, 21))
    assert "A1. 3月14日 对方去漫展见阿澪" in llm.prompts[0]
    again = AgendaBook(tmp_path / "agenda.json")
    assert again.items(KEY)[0].status == "done"


# --- what she is shown --------------------------------------------------------

def test_the_opener_is_told_which_day_it_is_for(tmp_path):
    book = _book(tmp_path, (HIS, "对方去漫展见阿澪", SHOW))
    [item] = book.open_items(KEY, date(2026, 3, 13))
    block, sid = ag.opener_block(item, datetime(2026, 3, 13, 11))
    assert sid == "A1" and "3月14日" in block and "明天" in block


def test_her_replies_see_what_is_ahead(tmp_path):
    book = _book(tmp_path, (HIS, "对方去漫展见阿澪", SHOW),
                 (HERS, "替他排队买团子周边", SHOW))
    block = ag.chat_block(book.open_items(KEY, date(2026, 3, 5)), date(2026, 3, 5))
    assert "对方去漫展见阿澪（他说过的）" in block and "替他排队买团子周边（你答应他的）" in block


# --- the opener path ---------------------------------------------------------

@pytest.mark.asyncio
async def test_an_opener_that_takes_up_the_item_marks_it(tmp_path):
    from lingxi.channels.outbound import ChannelRegistry, OutboundChannel
    from lingxi.temporal.proactive import ProactiveConfig, ProactiveScheduler
    from lingxi.temporal.tracker import InteractionTracker

    class _Ch(OutboundChannel):
        channel_name = "feishu"

        async def send_message(self, recipient_id, text, turn_id=None):
            pass

    book = _book(tmp_path, (HIS, "对方去漫展见阿澪", SHOW))
    tracker = InteractionTracker(tmp_path)
    rec = tracker.record_interaction("feishu", "oc_x")
    rec.last_interaction = datetime.now() - timedelta(days=2)
    reg = ChannelRegistry()
    reg.register(_Ch())

    async def _append(*a):
        pass

    engine = SimpleNamespace(
        annotation_store=None, _pending_stickers={}, agenda=book,
        memory=SimpleNamespace(embedding_provider=None,
                               short_term=SimpleNamespace(append_for_recipient=_append)))
    sched = ProactiveScheduler(config=ProactiveConfig(), tracker=tracker,
                               channel_registry=reg, engine=engine)
    now = datetime.now()
    [item] = book.open_items(KEY, date(2026, 3, 13))
    sched._agenda_offered = {KEY: (item, "before")}

    async def _ask(record, silence, now, force=False):
        return {"should_send": True, "message": "明天漫展了 你东西收拾好没",
                "sticker": "", "reason": "", "agenda": "A1"}

    sched._ask_llm = _ask
    result = await sched._maybe_reach_out(rec, now, force=True)
    assert result["status"] == "sent"
    asked = AgendaBook(tmp_path / "agenda.json").items(KEY)[0].asked
    assert [a["stage"] for a in asked] == ["before"]
    assert datetime.fromisoformat(asked[0]["at"]) == rec.proactive_outcomes[-1].sent_at
