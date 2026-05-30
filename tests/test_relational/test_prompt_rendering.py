"""Test that relational memory no longer renders into the system prompt.

Phase 3 context refactor — _build_relational_section was removed from
prompt_builder. Relational data now flows through brain/renderer 【你和他】
block (FactStore user:* facts) on the _prepare_turn_v2 path. The old
RelationalMemory object param is silently ignored.
"""

from lingxi.persona.models import Identity, PersonaConfig
from lingxi.persona.prompt_builder import PromptBuilder
from lingxi.relational.models import RelationalMemory


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

    def test_populated_memory_also_not_rendered(self):
        # Relational section removed — even populated RelationalMemory is silently
        # ignored. Rendering is now brain/renderer's job via 【你和他】.
        m = RelationalMemory(recipient_key="x", pet_names=["笨蛋"])
        prompt = PromptBuilder(_persona()).build_system_prompt(relational_memory=m)
        assert "我们」的部分" not in prompt
        assert "💞" not in prompt
