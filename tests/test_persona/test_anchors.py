"""Dated milestones render an elapsed span computed at prompt time.

Regression: the persona's background said 已经五年 in prose. True when written,
wrong the following April, and nothing would have flagged it.
"""
from datetime import date, datetime, timedelta

import yaml

from lingxi.persona.models import PersonaConfig
from lingxi.persona.prompt_builder import PromptBuilder


def _persona(anchors):
    return PersonaConfig(
        name="T", id="t", identity={"full_name": "T"}, anchors=anchors)


def _section(anchors):
    return PromptBuilder(_persona(anchors))._build_identity_section()


def test_elapsed_years_are_computed_from_today():
    five_ago = date.today().replace(year=date.today().year - 5)
    out = _section([{"event": "出道", "date": five_ago.isoformat()}])
    assert "到现在 **5 年**" in out
    assert f"{five_ago.year}年{five_ago.month}月" in out


def test_anniversary_not_yet_reached_counts_the_lower_year():
    # Same month/day next week → the year has not turned over yet.
    d = (datetime.now() + timedelta(days=7)).date().replace(
        year=datetime.now().year - 3)
    out = _section([{"event": "出道", "date": d.isoformat()}])
    assert "到现在 **2 年**" in out


def test_recent_anchor_falls_back_to_months():
    d = (datetime.now() - timedelta(days=70)).date()
    out = _section([{"event": "开始", "date": d.isoformat()}])
    assert "个月" in out


def test_note_is_appended_and_bad_dates_are_skipped():
    out = _section([
        {"event": "出道", "date": "2021-04-07", "note": "首张单曲"},
        {"event": "坏数据", "date": "not-a-date"},
    ])
    assert "首张单曲" in out
    assert "坏数据" not in out          # unparseable entries drop quietly


def test_tangkeke_debut_anchor_is_2021():
    p = PersonaConfig(**yaml.safe_load(open("config/personas/tangkeke.yaml")))
    debut = next(a for a in p.anchors if "出道" in a.event)
    assert debut.date.startswith("2021-04")


def test_age_is_computed_from_birthdate():
    from lingxi.persona.models import Identity
    born = date.today().replace(year=date.today().year - 21)
    assert Identity(full_name="T", birthdate=born.isoformat()).current_age() == 21
    # birthday still ahead this year → one year younger
    later = (datetime.now() + timedelta(days=10)).date().replace(
        year=datetime.now().year - 21)
    assert Identity(full_name="T", birthdate=later.isoformat()).current_age() == 20
    # no birthdate → the fixed age stands; bad birthdate falls back to it
    assert Identity(full_name="T", age=28).current_age() == 28
    assert Identity(full_name="T", birthdate="nope", age=28).current_age() == 28


def test_tangkeke_age_tracks_her_debut_era():
    p = PersonaConfig(**yaml.safe_load(open("config/personas/tangkeke.yaml")))
    age = p.identity.current_age()
    # 16 at the 2021 debut, so five-plus years on she is in her twenties —
    # the number moves with the calendar instead of being pinned at 16.
    assert age is not None and age >= 21
    assert "高中" not in (p.identity.occupation or "")
