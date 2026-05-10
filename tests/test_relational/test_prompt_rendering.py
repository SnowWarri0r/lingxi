"""Test that relational memory renders into the system prompt correctly."""

from datetime import datetime

from lingxi.persona.models import Identity, PersonaConfig
from lingxi.persona.prompt_builder import PromptBuilder
from lingxi.relational.models import (
    DailyPattern,
    FightPattern,
    InsideJoke,
    RelationalMemory,
    SharedPlace,
    SweetMoment,
)


def _persona():
    return PersonaConfig(name="T", identity=Identity(full_name="T"))


class TestEmptyMemoryNotRendered:
    def test_none_no_section(self):
        prompt = PromptBuilder(_persona()).build_system_prompt(relational_memory=None)
        assert "我们」的部分" not in prompt
        assert "💞" not in prompt

    def test_empty_memory_no_section(self):
        m = RelationalMemory(recipient_key="x")
        prompt = PromptBuilder(_persona()).build_system_prompt(relational_memory=m)
        assert "我们」的部分" not in prompt


class TestPopulatedRendering:
    def test_pet_names_appear_quoted(self):
        m = RelationalMemory(recipient_key="x", pet_names=["笨蛋", "老李"])
        prompt = PromptBuilder(_persona()).build_system_prompt(relational_memory=m)
        assert '"笨蛋"' in prompt
        assert '"老李"' in prompt

    def test_inside_jokes_render_with_origin(self):
        m = RelationalMemory(
            recipient_key="x",
            inside_jokes=[
                InsideJoke(phrase="蜘蛛会做梦的", origin="某次她梦到蜘蛛"),
            ],
        )
        prompt = PromptBuilder(_persona()).build_system_prompt(relational_memory=m)
        assert "蜘蛛会做梦的" in prompt
        assert "某次她梦到蜘蛛" in prompt

    def test_shared_place_renders(self):
        m = RelationalMemory(
            recipient_key="x",
            shared_places=[
                SharedPlace(name="楼下便利店", significance="他下班路过"),
            ],
        )
        prompt = PromptBuilder(_persona()).build_system_prompt(relational_memory=m)
        assert "楼下便利店" in prompt
        assert "他下班路过" in prompt

    def test_fight_pattern_renders_three_phases(self):
        m = RelationalMemory(
            recipient_key="x",
            fight_patterns=[
                FightPattern(
                    trigger="他不回消息",
                    her_pattern="嘴硬几小时",
                    typical_repair="他用具体小事示弱",
                ),
            ],
        )
        prompt = PromptBuilder(_persona()).build_system_prompt(relational_memory=m)
        assert "他不回消息" in prompt
        assert "嘴硬几小时" in prompt
        assert "示弱" in prompt

    def test_sweet_moment_renders_content(self):
        m = RelationalMemory(
            recipient_key="x",
            sweet_moments=[
                SweetMoment(
                    timestamp=datetime(2026, 5, 1),
                    content="他凌晨2点说想看流星雨",
                    weight="high",
                ),
            ],
        )
        prompt = PromptBuilder(_persona()).build_system_prompt(relational_memory=m)
        assert "他凌晨2点说想看流星雨" in prompt

    def test_daily_pattern_renders(self):
        m = RelationalMemory(
            recipient_key="x",
            daily_patterns=[DailyPattern(pattern="他每天 11 点下班")],
        )
        prompt = PromptBuilder(_persona()).build_system_prompt(relational_memory=m)
        assert "11 点下班" in prompt

    def test_relationship_summary_renders(self):
        m = RelationalMemory(
            recipient_key="x",
            relationship_summary="温和但带点距离感的朋友",
        )
        prompt = PromptBuilder(_persona()).build_system_prompt(relational_memory=m)
        assert "温和但带点距离感" in prompt


class TestSignaturePhrases:
    def test_signature_phrases_rendered_quoted(self):
        m = RelationalMemory(
            recipient_key="x",
            signature_phrases=["懂", "等下"],
        )
        prompt = PromptBuilder(_persona()).build_system_prompt(relational_memory=m)
        assert '"懂"' in prompt
        assert '"等下"' in prompt
        assert "你跟这个人说话长出来的口头" in prompt

    def test_signature_phrases_capped_at_six(self):
        m = RelationalMemory(
            recipient_key="x",
            signature_phrases=[f"p{i}" for i in range(15)],
        )
        prompt = PromptBuilder(_persona()).build_system_prompt(relational_memory=m)
        assert '"p5"' in prompt
        assert '"p6"' not in prompt

    def test_empty_signature_phrases_no_render(self):
        m = RelationalMemory(recipient_key="x", pet_names=["x"])  # has other content
        prompt = PromptBuilder(_persona()).build_system_prompt(relational_memory=m)
        assert "你跟这个人说话长出来的口头" not in prompt


class TestCaps:
    def test_inside_jokes_capped_at_six(self):
        m = RelationalMemory(
            recipient_key="x",
            inside_jokes=[
                InsideJoke(phrase=f"phrase{i}", origin="x")
                for i in range(20)
            ],
        )
        prompt = PromptBuilder(_persona()).build_system_prompt(relational_memory=m)
        assert "phrase0" in prompt
        assert "phrase5" in prompt
        assert "phrase6" not in prompt

    def test_pet_names_capped_at_five(self):
        m = RelationalMemory(
            recipient_key="x",
            pet_names=[f"name{i}" for i in range(10)],
        )
        prompt = PromptBuilder(_persona()).build_system_prompt(relational_memory=m)
        assert "name0" in prompt
        assert "name4" in prompt
        assert "name5" not in prompt
