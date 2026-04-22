"""Unit test for the annotation-button card element builder."""

from lingxi.channels.feishu import build_annotation_footer_elements


def test_footer_has_three_buttons_with_turn_id():
    elements = build_annotation_footer_elements(turn_id="abc123")
    # Should be a list with at least one action element containing 3 buttons
    assert isinstance(elements, list)
    flat = str(elements)
    assert "abc123" in flat
    # All three action values should be referenced
    assert "annotate_positive" in flat
    assert "annotate_negative" in flat
    assert "annotate_correction" in flat
