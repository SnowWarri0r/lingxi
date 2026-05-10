"""Tests for response cleaner — em-dash strip, narration removal, bubble split."""

from lingxi.conversation.response_cleaner import clean_speech, split_into_bubbles


class TestCleanSpeech:
    def test_strips_em_dash_double(self):
        assert "——" not in clean_speech("我懂——那种感觉")

    def test_strips_em_dash_spaced_single(self):
        assert " — " not in clean_speech("我懂 — 那种感觉")

    def test_preserves_double_newline_for_bubble_split(self):
        cleaned = clean_speech("第一句\n\n第二句\n\n第三句")
        assert "\n\n" in cleaned

    def test_collapses_three_plus_newlines(self):
        cleaned = clean_speech("第一句\n\n\n\n第二句")
        assert "\n\n\n" not in cleaned
        assert "\n\n" in cleaned


class TestSplitIntoBubbles:
    def test_single_message_returns_one_bubble(self):
        assert split_into_bubbles("一句话") == ["一句话"]

    def test_two_paragraphs_split(self):
        assert split_into_bubbles("第一句\n\n第二句") == ["第一句", "第二句"]

    def test_three_paragraphs_split(self):
        result = split_into_bubbles("a\n\nb\n\nc")
        assert result == ["a", "b", "c"]

    def test_caps_at_max_bubbles_by_merging_overflow(self):
        # Cap limits message COUNT but preserves all content — overflow
        # gets merged into the last allowed bubble, not dropped.
        result = split_into_bubbles("a\n\nb\n\nc\n\nd\n\ne", max_bubbles=3)
        assert len(result) == 3
        assert result[0] == "a"
        assert result[1] == "b"
        # Last bubble has c, d, e merged
        assert "c" in result[2] and "d" in result[2] and "e" in result[2]

    def test_under_cap_unchanged(self):
        # Equal to cap: no merging
        result = split_into_bubbles("a\n\nb\n\nc", max_bubbles=3)
        assert result == ["a", "b", "c"]

    def test_drops_empty_pieces(self):
        # Stray empty segments don't become bubbles
        result = split_into_bubbles("a\n\n\n\nb")  # cleaned input usually wouldn't, but just in case
        assert "" not in result
        assert result == ["a", "b"]

    def test_empty_input_returns_empty_list(self):
        assert split_into_bubbles("") == []

    def test_strips_per_bubble(self):
        result = split_into_bubbles("  hello  \n\n  world  ")
        assert result == ["hello", "world"]
