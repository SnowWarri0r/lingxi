"""The weekly report groups openers by week, with kinds and replies."""

import importlib.util
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("weekly_report", ROOT / "scripts" / "weekly_report.py")
wr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wr)


def test_openers_are_counted_by_the_monday_of_their_week():
    records = [{"channel": "feishu", "recipient_id": "oc_x", "proactive_outcomes": [
        {"sent_at": "2026-03-09T11:00:00", "answered": True, "aspect": "队友"},
        {"sent_at": "2026-03-15T16:00:00", "answered": False, "aspect": "练习排练"},
        {"sent_at": "2026-03-16T11:00:00", "answered": False}]}]
    rows = wr.opener_rows(records, "feishu:oc_x")
    assert rows["2026-03-09"]["sent"] == 2 and rows["2026-03-09"]["answered"] == 1
    assert rows["2026-03-09"]["kinds"] == Counter({"队友": 1, "练习排练": 1})
    assert rows["2026-03-16"]["kinds"] == Counter({"（未标）": 1})


def test_her_voice_is_measured_per_week():
    mine = ["可可来啦！", "嗯 好"] * 5
    turns = [{"role": "assistant", "timestamp": f"2026-03-1{i % 5}T20:00:00",
              "content": c} for i, c in enumerate(mine)]
    turns.append({"role": "user", "timestamp": "2026-03-11T20:01:00", "content": "可可！"})
    v = wr.voice_rows(turns, "可可")["2026-03-09"]
    assert v.n == 10 and v.exclaim == 0.5 and v.self_name == 0.5
