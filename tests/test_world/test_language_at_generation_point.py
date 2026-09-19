"""The language requirement has to sit where the field is filled in.

The directive lived mid-prompt with 475 characters of examples and schema
after it, and the last thing the model read before writing the item was
`"voice": "她语气的一句（<= 50 字）"` — no mention of language. Yield tells the
story: once the script filter started rejecting foreign items, items written
per day went 4, 4, 4, 4 → 2, 1, because the fetch kept producing the source's
language and the filter kept throwing it away.

The filter stays as the backstop; this is the generation-side half.
"""

from datetime import date

from lingxi.persona.models import Identity, PersonaConfig
from lingxi.world.fetcher import build_fetch_prompt


def _persona():
    return PersonaConfig(
        name="唐可可", id="tangkeke",
        identity=Identity(full_name="唐可可", age=21),
        world_interests=["日本的偶像/动画圈动向"])


def _voice_field_line(prompt: str) -> str:
    return next(ln for ln in prompt.splitlines() if '"voice"' in ln)


def test_the_voice_field_itself_states_the_language():
    line = _voice_field_line(build_fetch_prompt(_persona(), date(2026, 9, 13)))

    assert "语言" in line


def test_the_field_says_not_the_sources_language():
    line = _voice_field_line(build_fetch_prompt(_persona(), date(2026, 9, 13)))

    assert "原文" in line


def test_the_earlier_directive_is_still_there():
    """Two mentions, not one moved: the mid-prompt one carries the reason."""
    prompt = build_fetch_prompt(_persona(), date(2026, 9, 13))

    assert prompt.count("语言") >= 3


def test_the_field_description_still_carries_the_length_bound():
    line = _voice_field_line(build_fetch_prompt(_persona(), date(2026, 9, 13)))

    assert "50" in line
