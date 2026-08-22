"""Proactive messages carry an age, and near-duplicates don't go out.

Regression: history was a bare list of strings rendered without timestamps, so
she could not tell she had said a line yesterday — and 换一件事说 did not stop
her re-sending one almost word for word. A topic then stayed alive for days on
the strength of her own repeats.
"""
from datetime import datetime, timedelta

from lingxi.temporal.proactive import _ago_label, _as_entry, _too_similar


NOW = datetime(2026, 8, 16, 10, 0)


def test_ago_label_spells_out_the_gap():
    assert "今天" in _ago_label((NOW - timedelta(hours=2)).isoformat(), NOW)
    assert _ago_label((NOW - timedelta(days=1)).isoformat(), NOW) == "昨天"
    assert _ago_label((NOW - timedelta(days=3)).isoformat(), NOW) == "3天前"
    # legacy rows carry no timestamp and must still render
    assert _ago_label(None) == "更早"
    assert _ago_label("garbage") == "更早"


def test_entries_accept_both_storage_formats():
    assert _as_entry("旧字符串") == {"text": "旧字符串", "ts": None}
    e = _as_entry({"text": "新格式", "ts": "2026-08-16T10:00:00"})
    assert e["text"] == "新格式" and e["ts"] == "2026-08-16T10:00:00"


def test_near_verbatim_repeat_is_caught():
    # The pair that actually shipped a day apart.
    a = "刚洗完澡 头发还湿着坐床边\n\n突然想通一件事 日记不念给你听也没关系 但是想告诉你的话 我直接说就好啦 嘿嘿"
    b = "刚洗完澡头发还湿着 坐床边突然想到\n\n日记不念给你听也没关系 但想告诉你的话 我直接说就好啦 嘿嘿"
    assert _too_similar(b, [a]) == a


def test_different_messages_pass():
    prev = ["刚洗完澡头发还湿着 坐床边突然想到 日记的事"]
    assert _too_similar("刚剥个橙子剥得满手黏 我妈还在视频里说多吃萝卜", prev) is None
    assert _too_similar("今天练到腿都软了", prev) is None


def test_very_short_messages_are_not_deduped():
    # "早呀" repeating is fine; the guard targets whole repeated openers.
    assert _too_similar("早呀", ["早呀"]) is None
