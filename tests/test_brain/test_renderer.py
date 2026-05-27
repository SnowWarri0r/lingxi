from lingxi.persona.loader import load_persona
from lingxi.persona.prompt_builder import build_persona_block


def test_persona_block_extracts_static_sections():
    persona = load_persona("config/personas/example_persona.yaml")
    block = build_persona_block(persona)
    # Persona name should appear (either via 'Aria' or persona.name)
    assert "Aria" in block or persona.name in block
    # The "how to talk" rules section should be in the format preamble
    assert "## 怎么说话" in block
    # The output format marker should appear
    assert "===META===" in block
