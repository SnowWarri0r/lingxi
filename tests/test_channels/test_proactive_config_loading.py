"""Every proactive knob in the YAML has to reach the object.

The loader listed its keys by hand and two of them were never on the list:
reengage_backoff and reengage_max_hours. Setting either in config/default.yaml
did nothing and said nothing — the code default applied while the file
claimed otherwise. That is how the doubling stayed at 2.0x, 14-day ceiling
included, no matter what the file said.
"""

from lingxi.channels.feishu_cli import load_proactive_config
from lingxi.temporal.proactive import ProactiveConfig


def test_the_two_keys_that_were_silently_ignored_now_land():
    cfg = {"proactive": {"reengage_backoff": 1.0, "reengage_max_hours": 48.0}}

    out = load_proactive_config(cfg)

    assert out.reengage_backoff == 1.0
    assert out.reengage_max_hours == 48.0


def test_no_backoff_means_a_flat_wait():
    cfg = {"proactive": {"reengage_after_hours": 6, "reengage_backoff": 1.0}}

    out = load_proactive_config(cfg)
    waits = [out.reengage_wait_for(n).total_seconds() / 3600 for n in range(2, 25)]

    assert set(waits) == {6.0}


def test_an_empty_config_gives_the_model_defaults():
    out = load_proactive_config({})
    default = ProactiveConfig()

    assert out.reengage_backoff == default.reengage_backoff
    assert out.cooldown_hours == default.cooldown_hours


def test_silence_thresholds_keep_integer_keys():
    """YAML can hand these back as strings; the lookup is by int level."""
    cfg = {"proactive": {"silence_thresholds": {"1": 5, "2": 4}}}

    out = load_proactive_config(cfg)

    assert out.silence_threshold_for(2).total_seconds() / 3600 == 4


def test_an_unknown_key_does_not_explode():
    out = load_proactive_config({"proactive": {"not_a_field": 1, "cooldown_hours": 3}})

    assert out.cooldown_hours == 3


def test_every_field_is_reachable_from_yaml():
    """The guard against this regressing: no field may be unreachable."""
    every = {name: getattr(ProactiveConfig(), name)
             for name in ProactiveConfig.model_fields}
    every["cooldown_hours"] = 99.0
    every["reengage_backoff"] = 1.25
    every["reengage_max_hours"] = 7.0
    every["max_consecutive_proactive"] = 9

    out = load_proactive_config({"proactive": every})

    assert out.cooldown_hours == 99.0
    assert out.reengage_backoff == 1.25
    assert out.reengage_max_hours == 7.0
    assert out.max_consecutive_proactive == 9
