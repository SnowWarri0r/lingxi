"""Openers rotate across sides of her life in code, not by request.

Asked to pick another aspect, she still picked practice: 9 of 14 openers
after 10-10 were rehearsal details, with news and teammate stories beside
them on the shelf.
"""

from lingxi.planner.stories import Story
from lingxi.temporal.proactive import rotate_stories

SHELF = [Story("a", "压腿压到腿后侧发紧", "练习排练"),
         Story("b", "新番官宣动画化了", "圈里新鲜事"),
         Story("c", "菫又来抢我可乐", "队友")]


def test_the_aspects_of_the_last_two_openers_are_held_back():
    recent = [{"aspect": "练习排练"}, {"aspect": "队友"}]
    assert [s.id for s in rotate_stories(SHELF, recent)] == ["b"]


def test_an_older_opener_does_not_hold_anything_back():
    recent = [{"aspect": "练习排练"}, {"aspect": "天气"}, {"aspect": "吃喝"}]
    assert [s.id for s in rotate_stories(SHELF, recent)] == ["a", "b", "c"]


def test_when_everything_was_just_used_she_still_has_something():
    recent = [{"aspect": "练习排练"}, {"aspect": "练习排练"}]
    only = [SHELF[0]]
    assert rotate_stories(only, recent) == only
