"""Proactive tone/volume guidance must reach the REAL scheduled path.

Regression: the style list (deliberately ordinary-volume examples) used to
hang only off the `force` test branch, so the actually-scheduled message got
no register guidance and defaulted to the persona's peak energy — every
opener came out at maximum volume, all ending in '！'.
"""
import inspect
from datetime import datetime

from lingxi.temporal import proactive
from lingxi.temporal.proactive import _MESSAGE_STYLES, _format_own_life_block
from lingxi.facts.models import Fact, FactType, Source


def test_style_block_is_built_before_the_force_branch():
    """style_block must be computed unconditionally, then used in BOTH paths."""
    src = inspect.getsource(proactive.ProactiveScheduler)
    assert "style_block = (" in src
    before_branch, _, after_branch = src.partition("if force:")
    # built before the branch (so both paths see it)
    assert "style_block = (" in before_branch
    # and referenced at least twice after — once per path
    assert after_branch.count("{style_block}") >= 2


def test_all_style_examples_are_ordinary_volume():
    """The examples supply the missing middle register, so they stay calm."""
    for s in _MESSAGE_STYLES:
        assert s["example"], s["name"]
        assert "！" not in s["example"] and "!" not in s["example"], s["name"]


def test_opener_shape_states_ordinary_volume_as_the_default():
    src = inspect.getsource(proactive.ProactiveScheduler)
    assert "音量按事情本身来" in src
    assert "平常音量" in src


def test_own_life_block_allows_ordinary_material():
    facts = [Fact(subject="aria", content="下午把葱油熬好了",
                  type=FactType.EVENT, source=Source.LIFE_SIMULATED,
                  ts=datetime(2026, 7, 28, 15, 0))]
    block = _format_own_life_block(facts)
    # ordinary days are explicitly valid opener material
    assert "平平常常" in block


def test_follow_up_styles_do_not_assume_the_topic_is_the_users():
    """Regression: 话题跟进 read as 'follow up on something THEY told you', so
    when the live thread was her own diary she asked the user how *their* diary
    was going."""
    styles = {s["name"]: s["desc"] for s in _MESSAGE_STYLES}
    assert "先认清那件事是谁的" in styles["话题跟进"]
    # 关心 stays user-directed but says what to do when there's nothing of theirs
    assert "对方自己说过的" in styles["关心"]
    assert "换一件你自己的事说" in styles["关心"]


def test_opener_shape_requires_checking_whose_topic_it_is():
    src = inspect.getsource(proactive.ProactiveScheduler)
    assert "提旧事先认主语" in src
