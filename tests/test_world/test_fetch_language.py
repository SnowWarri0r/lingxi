"""The item has to come back in the language she speaks, not the source's.

Nothing in the fetch prompt said what language to write in. On the first day
the sources were Chinese and so was the output; on the second, a category
covering her industry (which is Japanese) pulled Japanese sources and all four
items came back in Japanese — for a persona who talks to him in Chinese, and
straight into the block that gets pushed into her turn.
"""

from datetime import date

from lingxi.persona.models import Identity, PersonaConfig
from lingxi.world.fetcher import build_fetch_prompt


def _persona(**kw):
    return PersonaConfig(
        name="唐可可", id="tangkeke",
        identity=Identity(full_name="唐可可", age=21),
        world_interests=["日本的偶像/动画圈动向"],
        **kw)


def test_the_prompt_pins_the_output_language_to_her_own():
    prompt = build_fetch_prompt(_persona(), date(2026, 9, 9))

    assert "语言" in prompt


def test_it_points_at_the_persona_rather_than_naming_a_language():
    """Hardcoding 中文 here is the same mistake as hardcoding her interests."""
    prompt = build_fetch_prompt(_persona(), date(2026, 9, 9))
    directive = prompt[prompt.index("语言") - 120:prompt.index("语言") + 120]

    assert "日文" not in directive and "日本語" not in directive


def test_the_source_language_is_called_out_as_the_thing_not_to_follow():
    prompt = build_fetch_prompt(_persona(), date(2026, 9, 9))

    assert "原文" in prompt
