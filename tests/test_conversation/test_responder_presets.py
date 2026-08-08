"""Responder selection: preset table drives which model speaks to the user.

The doubao wiring used to be hardcoded in _get_responder_llm; adding a second
vendor (DeepSeek) turned it into a table. These pin the resolution rules —
env vs yaml precedence, per-vendor defaults, and the fall-back-to-main path
that keeps chat alive when a key is absent.
"""
import pytest

from lingxi.conversation.engine import ConversationEngine, RESPONDER_PRESETS
from lingxi.memory.manager import MemoryManager
from lingxi.providers.openai_provider import OpenAIProvider


def _engine(persona, llm, tmp_path):
    return ConversationEngine(
        persona=persona, llm_provider=llm,
        memory_manager=MemoryManager(data_dir=str(tmp_path / "memory")),
    )


def test_presets_cover_both_vendors_with_required_fields():
    assert {"doubao", "deepseek"} <= set(RESPONDER_PRESETS)
    for name, p in RESPONDER_PRESETS.items():
        assert p["key_env"] and p["model_env"] and p["base_url"], name


def test_deepseek_preset_points_at_v4_flash():
    p = RESPONDER_PRESETS["deepseek"]
    assert p["base_url"] == "https://api.deepseek.com"
    assert p["default_model"] == "deepseek-v4-flash"
    # Thinking defaults to ON (effort=high) on DeepSeek, so it must be turned
    # off explicitly or every reply pays a reasoning pass before the first token.
    assert p["extra_body"] == {"thinking": {"type": "disabled"}}


def test_deepseek_builds_openai_provider_from_env(
        sample_persona, mock_llm, tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.delenv("DEEPSEEK_RESPONDER_MODEL", raising=False)
    sample_persona.responder.provider = "deepseek"
    sample_persona.responder.model = ""
    eng = _engine(sample_persona, mock_llm, tmp_path)

    llm = eng._get_responder_llm()
    assert isinstance(llm, OpenAIProvider)
    assert llm.model == "deepseek-v4-flash"        # preset default applies
    assert llm._base_url == "https://api.deepseek.com"
    assert eng._responder_is_external() is True    # single pass, no chat tools


def test_yaml_model_overrides_env(sample_persona, mock_llm, tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setenv("DEEPSEEK_RESPONDER_MODEL", "from-env")
    sample_persona.responder.provider = "deepseek"
    sample_persona.responder.model = "from-yaml"
    llm = _engine(sample_persona, mock_llm, tmp_path)._get_responder_llm()
    assert llm.model == "from-yaml"


def test_missing_key_falls_back_to_main_llm(
        sample_persona, mock_llm, tmp_path, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    sample_persona.responder.provider = "deepseek"
    eng = _engine(sample_persona, mock_llm, tmp_path)
    assert eng._get_responder_llm() is mock_llm    # chat still works


def test_doubao_preset_keeps_thinking_disabled():
    # Regression: losing this makes doubao sit ~15s before the first token.
    assert RESPONDER_PRESETS["doubao"]["extra_body"] == {
        "thinking": {"type": "disabled"}}
