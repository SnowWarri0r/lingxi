"""She is told when her typing drifts from the persona's band.

September held a persona whose top trait is 元气 at zero '!' in 51 messages,
locked there by her own flat history, and nothing noticed. Replayed on those
turns the monitor fired on 11 of 11; the line it adds doubled her using her
own name (8 → 16 of 33) but moved '!' only from 1 to 2 — her history still
outweighs one line. It is a watch with a nudge, not a cure.
"""

from types import SimpleNamespace

from lingxi.persona.voice_monitor import (
    MIN_MESSAGES, WINDOW, correction, measure, voice_block)

BAND = {"exclaim_min": 0.15, "shouting_max": 0.25, "self_name": "可可",
        "self_name_min": 0.1, "self_name_note": "说到自己常用「可可」自称。"}
FLAT = ["今天练了一下午 腿酸"] * 10
LOUD = ["啊啊啊好棒！！！"] * 10
OWN = ["可可今天练完了！好累"] * 3 + ["今天下雨了"] * 7


def test_too_few_messages_are_not_judged():
    assert measure(FLAT[:MIN_MESSAGES - 1], "可可") is None


def test_only_the_latest_window_counts():
    stats = measure(LOUD * 3 + FLAT * 2, "可可")
    assert stats.n == WINDOW and stats.exclaim == 0


def test_a_flat_run_draws_the_quiet_note():
    block = correction(measure(FLAT, "可可"), BAND)
    assert "收着" in block and "「可可」" in block


def test_a_shouting_run_draws_the_loud_note():
    block = correction(measure(LOUD, "可可"), BAND)
    assert "吵" in block


def test_in_band_says_nothing():
    assert correction(measure(OWN, "可可"), BAND) is None


def test_a_persona_without_a_band_is_never_checked():
    assert voice_block(SimpleNamespace(voice_baseline={}), FLAT) is None


def test_the_live_persona_declares_a_band():
    from lingxi.persona.loader import load_persona
    p = load_persona("config/personas/tangkeke.yaml")
    assert p.voice_baseline.get("exclaim_min")
