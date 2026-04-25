"""Tests for FewShotSample and AnnotationTurn."""

from datetime import datetime

import pytest

from lingxi.fewshot.models import AnnotationTurn, FewShotSample


class TestFewShotSample:
    def test_defaults(self):
        s = FewShotSample(
            id="1",
            inner_thought="tired",
            corrected_speech="累。",
            context_summary="late night",
        )
        assert s.source == "seed"
        assert s.original_speech is None
        assert s.tags == []
        assert s.recipient_key is None
        assert isinstance(s.created_at, datetime)

    def test_source_values(self):
        for src in ("seed", "user_correction", "positive"):
            FewShotSample(
                id=src, inner_thought="x", corrected_speech="y",
                context_summary="z", source=src,
            )

    def test_invalid_source_rejected(self):
        with pytest.raises(Exception):
            FewShotSample(
                id="x", inner_thought="a", corrected_speech="b",
                context_summary="c", source="bogus",  # type: ignore
            )


class TestAnnotationTurn:
    def test_defaults(self):
        t = AnnotationTurn(
            turn_id="t1",
            recipient_key="cli:me",
            user_message="hi",
            inner_thought="",
            speech="hello",
        )
        assert t.annotation == "none"
        assert t.correction is None

    def test_set_correction(self):
        t = AnnotationTurn(
            turn_id="t1", recipient_key="cli:me",
            user_message="hi", inner_thought="", speech="hello",
        )
        t.annotation = "negative"
        t.correction = "嗨"
        assert t.correction == "嗨"
