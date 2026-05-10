"""Tests for response cleaner — em-dash strip, narration removal, bubble split."""

from lingxi.conversation.response_cleaner import clean_speech, split_into_bubbles


class TestMonologueLeak:
    """Regression tests — internal-thought narration leaking into speech."""

    def test_drops_production_leak_text(self):
        # The actual line that shipped: Aria's retrospective monologue
        # got rendered as a chat bubble.
        leaked = "想起来昨天他问我最近都在干什么，还问我有没有瞒着什么好玩的事..."
        cleaned = clean_speech(leaked)
        assert "想起来" not in cleaned
        assert "他问我" not in cleaned

    def test_drops_internal_thought_with_third_person_user(self):
        text = "我突然想起他昨天问我那个事"
        assert clean_speech(text).strip() == ""

    def test_drops_just_now_thinking_about_chat(self):
        text = "刚才在想她跟我说的那句话"
        assert clean_speech(text).strip() == ""

    def test_keeps_legit_retrospective_to_user(self):
        # "想起来" + "你" (talking to user) is fine — not a leak
        text = "想起来一件事 你猜怎么着"
        assert "想起来一件事" in clean_speech(text)

    def test_keeps_legit_third_person_when_addressing_user(self):
        # Aria talking ABOUT a third person, TO the user — has 你, keep
        text = "我妈今天问我你最近怎么样"
        cleaned = clean_speech(text)
        assert "我妈" in cleaned

    def test_keeps_bare_observation(self):
        # "我注意到" without third-person-to-user leak signal — keep
        text = "我注意到一个 bug"
        assert "我注意到" in clean_speech(text)

    def test_keeps_legit_chat_history_followup_with_you(self):
        # Codex P2: "想起来你上次提到了..." is direct follow-up, MUST keep
        text = "想起来你上次提到了那本书"
        assert "想起来" in clean_speech(text)
        assert "提到了" in clean_speech(text)

    def test_keeps_legit_chat_history_followup_with_we(self):
        # Codex P2: "想到我们之前聊过的电影" is joint reference, MUST keep
        text = "我突然想到我们之前聊过的那个电影"
        cleaned = clean_speech(text)
        assert "聊过" in cleaned
        assert "我们" in cleaned

    def test_keeps_咱们_followup(self):
        text = "刚才在想咱们聊起的那个事"
        assert "聊起" in clean_speech(text)


class TestAIPhraseBlacklist:
    def test_drops_ai_template_phrase_line(self):
        text = "嗯 好的\n希望这对你有帮助"
        cleaned = clean_speech(text)
        assert "希望这对你有帮助" not in cleaned
        assert "嗯 好的" in cleaned

    def test_drops_total_summary_phrase(self):
        text = "总的来说，这件事还是要慎重"
        assert "总的来说" not in clean_speech(text)

    def test_drops_great_question_phrase(self):
        text = "这是一个很好的问题，让我想想"
        assert "这是一个很好的问题" not in clean_speech(text)


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
