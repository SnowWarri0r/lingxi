"""Tests for prompt_builder._age_label — calendar-day-aware time labeling.

Regression: the old rolling-24h "今天早些时候" label let the LLM hallucinate
"早上看流星雨" because a yesterday-evening meteor-shower event got tagged
as "today early" — model interpreted as "this morning". The new logic
uses calendar-day comparison past 3h so 23:00 yesterday → "昨晚".
"""

from datetime import datetime

from lingxi.persona.prompt_builder import _age_label, _time_of_day_label


class TestTimeOfDayBucket:
    def test_凌晨(self):
        assert _time_of_day_label(0) == "凌晨"
        assert _time_of_day_label(5) == "凌晨"

    def test_上午(self):
        assert _time_of_day_label(6) == "上午"
        assert _time_of_day_label(10) == "上午"

    def test_中午(self):
        assert _time_of_day_label(11) == "中午"
        assert _time_of_day_label(13) == "中午"

    def test_下午(self):
        assert _time_of_day_label(14) == "下午"
        assert _time_of_day_label(17) == "下午"

    def test_晚上(self):
        assert _time_of_day_label(18) == "晚上"
        assert _time_of_day_label(23) == "晚上"


class TestAgeLabelRecency:
    def test_just_now_under_10min(self):
        now = datetime(2026, 5, 8, 13, 0)
        ts = datetime(2026, 5, 8, 12, 55)
        assert _age_label(ts, now) == "刚刚"

    def test_minutes_ago(self):
        now = datetime(2026, 5, 8, 13, 0)
        ts = datetime(2026, 5, 8, 12, 30)
        assert _age_label(ts, now) == "30分钟前"

    def test_within_3h_uses_hours(self):
        now = datetime(2026, 5, 8, 13, 0)
        ts = datetime(2026, 5, 8, 11, 0)
        assert _age_label(ts, now) == "2小时前"


class TestAgeLabelCalendarDay:
    def test_yesterday_evening_event_is_昨晚(self):
        # The meteor-shower bug: 14h ago at 23:00 yesterday must be "昨晚",
        # not "今天早些时候"
        now = datetime(2026, 5, 8, 13, 0)
        ts = datetime(2026, 5, 7, 23, 0)
        assert _age_label(ts, now) == "昨晚"

    def test_yesterday_afternoon_event(self):
        # 19h ago at 18:00 yesterday — also "昨晚" (18-23 falls in 晚上)
        now = datetime(2026, 5, 8, 13, 0)
        ts = datetime(2026, 5, 7, 18, 0)
        assert _age_label(ts, now) == "昨晚"

    def test_yesterday_midday_event(self):
        # 23h ago at 14:00 yesterday — "昨天下午"
        now = datetime(2026, 5, 8, 13, 0)
        ts = datetime(2026, 5, 7, 14, 0)
        assert _age_label(ts, now) == "昨天下午"

    def test_yesterday_morning_event(self):
        now = datetime(2026, 5, 8, 13, 0)
        ts = datetime(2026, 5, 7, 9, 0)
        # > 24h cutoff in caller, but label itself should still be correct
        assert _age_label(ts, now) == "昨天上午"

    def test_today_early_morning_event(self):
        # Same calendar day, event at 03:00 today (10h ago)
        now = datetime(2026, 5, 8, 13, 0)
        ts = datetime(2026, 5, 8, 3, 0)
        assert _age_label(ts, now) == "今天凌晨"

    def test_today_morning_event(self):
        now = datetime(2026, 5, 8, 17, 0)
        ts = datetime(2026, 5, 8, 8, 0)
        assert _age_label(ts, now) == "今天上午"

    def test_today_noon_event(self):
        now = datetime(2026, 5, 8, 17, 0)
        ts = datetime(2026, 5, 8, 12, 0)
        assert _age_label(ts, now) == "今天中午"

    def test_two_days_ago(self):
        now = datetime(2026, 5, 8, 13, 0)
        ts = datetime(2026, 5, 6, 18, 0)
        assert _age_label(ts, now) == "2天前"


class TestAgeLabelMidnightEdge:
    def test_just_after_midnight_recent_event(self):
        # User is awake at 02:00, event happened 22:00 yesterday (4h ago)
        # — within 3h cutoff... actually 4h, falls past. days_ago=1, tod=晚上
        # → "昨晚" reads naturally
        now = datetime(2026, 5, 8, 2, 0)
        ts = datetime(2026, 5, 7, 22, 0)
        assert _age_label(ts, now) == "昨晚"

    def test_just_after_midnight_very_recent(self):
        # 02:00 now, event at 23:30 yesterday (2.5h ago) — within 3h cutoff
        now = datetime(2026, 5, 8, 2, 0)
        ts = datetime(2026, 5, 7, 23, 30)
        assert _age_label(ts, now) == "2小时前"
