"""Tests for prompt_assembly helpers used by the single-call combo."""

import random

import pytest

from lingxi.conversation.prompt_assembly import (
    build_style_preamble,
    pick_prefill,
    render_fewshots_as_messages,
)
from lingxi.fewshot.models import FewShotSample
from lingxi.persona.models import StyleConfig


def _sample(context: str, speech: str) -> FewShotSample:
    return FewShotSample(
        id="x",
        inner_thought="...",
        original_speech=None,
        corrected_speech=speech,
        context_summary=context,
        tags=[],
        recipient_key=None,
        source="seed",
    )


class TestRenderFewshots:
    def test_empty_returns_empty_list(self):
        assert render_fewshots_as_messages([]) == []

    def test_one_pair(self):
        out = render_fewshots_as_messages([_sample("深夜加班", "累死了")])
        assert out == [
            {"role": "user", "content": "深夜加班"},
            {"role": "assistant", "content": "累死了"},
        ]

    def test_multiple_pairs_preserve_order(self):
        out = render_fewshots_as_messages([
            _sample("A", "a"),
            _sample("B", "b"),
        ])
        assert [m["content"] for m in out] == ["A", "a", "B", "b"]


class TestBuildStylePreamble:
    def test_includes_max_chars(self):
        cfg = StyleConfig(speech_max_chars=25)
        pre = build_style_preamble(cfg)
        assert "≤25" in pre or "25" in pre

    def test_default_blacklist_included(self):
        pre = build_style_preamble(StyleConfig())
        assert "希望" in pre
        assert "总是让人" in pre

    def test_persona_blacklist_appended(self):
        cfg = StyleConfig(blacklist_phrases=["据说"])
        pre = build_style_preamble(cfg)
        assert "据说" in pre

    def test_ends_with_newline(self):
        """Preamble must end with \\n so engine can concat with user message directly."""
        pre = build_style_preamble(StyleConfig())
        assert pre.endswith("\n")

    def test_engine_concat_produces_clean_separation(self):
        """Simulate engine concatenation (no extra newlines added by caller)."""
        pre = build_style_preamble(StyleConfig())
        wrapped = pre + "你今天吃啥"
        # User message immediately follows preamble
        assert wrapped.index("你今天吃啥") == len(pre)
        # And the style block is still at the start
        assert wrapped.startswith("[本轮")


class TestPickPrefill:
    def test_returns_value_from_openers(self):
        cfg = StyleConfig(prefill_openers=["嗯", "欸"])
        rng = random.Random(42)
        pick = pick_prefill(cfg, rng=rng)
        assert pick in ("嗯", "欸")

    def test_empty_list_returns_empty(self):
        cfg = StyleConfig(prefill_openers=[])
        assert pick_prefill(cfg) == ""

    def test_empty_string_option_allowed(self):
        cfg = StyleConfig(prefill_openers=[""])
        assert pick_prefill(cfg) == ""
