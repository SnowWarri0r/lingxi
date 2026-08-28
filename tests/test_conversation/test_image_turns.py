"""An image turn must survive a responder that cannot see images.

Live failure: a Feishu sticker reached deepseek-v4-flash as an image block,
the API answered `400 This model does not support image`, generation came
back empty and the user got the empty-reply fallback. Twice in one day.
"""

import pytest

from lingxi.conversation.engine import ConversationEngine
from lingxi.memory.manager import MemoryManager


def _engine(persona, llm, tmp_path, provider="deepseek"):
    persona.responder.provider = provider
    return ConversationEngine(
        persona=persona, llm_provider=llm,
        memory_manager=MemoryManager(data_dir=str(tmp_path / "memory")),
    )


def _image_blocks(messages):
    out = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            out += [b for b in content if b.get("type") == "image"]
    return out


def _last_user_text(messages):
    content = messages[-1]["content"]
    if isinstance(content, str):
        return content
    return " ".join(b.get("text", "") for b in content if b.get("type") == "text")


IMG = {"data": "AAAA", "media_type": "image/png"}


# --- which responders can read an image block -------------------------------

def test_deepseek_cannot_see_images(sample_persona, mock_llm, tmp_path):
    assert _engine(sample_persona, mock_llm, tmp_path)._responder_sees_images() is False


def test_doubao_can(sample_persona, mock_llm, tmp_path):
    eng = _engine(sample_persona, mock_llm, tmp_path, provider="doubao")
    assert eng._responder_sees_images() is True


def test_the_main_model_can(sample_persona, mock_llm, tmp_path):
    eng = _engine(sample_persona, mock_llm, tmp_path, provider="main")
    assert eng._responder_sees_images() is True


def test_an_unknown_provider_degrades_to_the_main_model_and_can(
        sample_persona, mock_llm, tmp_path):
    eng = _engine(sample_persona, mock_llm, tmp_path, provider="nonesuch")
    assert eng._responder_sees_images() is True


# --- what the turn looks like once the image has been described -------------

@pytest.mark.asyncio
async def test_no_image_block_reaches_a_text_only_responder(
        sample_persona, mock_llm, tmp_path):
    eng = _engine(sample_persona, mock_llm, tmp_path)
    _sys, messages = await eng._prepare_turn_v2(
        "", [IMG], "feishu", "oc_test")

    assert _image_blocks(messages) == []


@pytest.mark.asyncio
async def test_the_description_reaches_the_responder_as_text(
        sample_persona, tmp_path):
    from tests.conftest import MockLLMProvider

    eng = _engine(sample_persona, MockLLMProvider(["一只猫顶着锅盖"]), tmp_path)
    _sys, messages = await eng._prepare_turn_v2("", [IMG], "feishu", "oc_test")

    assert "一只猫顶着锅盖" in _last_user_text(messages)


@pytest.mark.asyncio
async def test_the_buffer_remembers_what_the_picture_was(
        sample_persona, tmp_path):
    """`[发送了1张图片]` alone left her unable to refer back to it a turn later."""
    from tests.conftest import MockLLMProvider

    eng = _engine(sample_persona, MockLLMProvider(["一只猫顶着锅盖"]), tmp_path)
    await eng._prepare_turn_v2("", [IMG], "feishu", "oc_test")

    history = eng.memory.short_term.get_history()
    assert "一只猫顶着锅盖" in history[-1].content


@pytest.mark.asyncio
async def test_the_users_own_words_survive_alongside_the_description(
        sample_persona, tmp_path):
    from tests.conftest import MockLLMProvider

    eng = _engine(sample_persona, MockLLMProvider(["海边的照片"]), tmp_path)
    _sys, messages = await eng._prepare_turn_v2(
        "你看这个", [IMG], "feishu", "oc_test")

    text = _last_user_text(messages)
    assert "你看这个" in text and "海边的照片" in text


@pytest.mark.asyncio
async def test_a_failed_look_still_produces_a_turn(sample_persona, tmp_path):
    class _Boom:
        async def complete(self, **kwargs):
            raise RuntimeError("vision endpoint down")

    eng = _engine(sample_persona, _Boom(), tmp_path)
    _sys, messages = await eng._prepare_turn_v2("", [IMG], "feishu", "oc_test")

    assert _image_blocks(messages) == []
    assert _last_user_text(messages).strip(), "an image turn is never empty text"


@pytest.mark.asyncio
async def test_a_vision_responder_keeps_the_image(sample_persona, tmp_path):
    from tests.conftest import MockLLMProvider

    eng = _engine(sample_persona, MockLLMProvider(["海边的照片"]), tmp_path,
                  provider="doubao")
    _sys, messages = await eng._prepare_turn_v2("", [IMG], "feishu", "oc_test")

    assert len(_image_blocks(messages)) == 1


@pytest.mark.asyncio
async def test_a_vision_responder_still_leaves_a_description_in_the_buffer(
        sample_persona, tmp_path):
    """History is text for every path.

    Seeing the image on this turn does nothing for the next one, or for the
    proactive path — both read the buffer, and a bare `[发送了1张图片]` there is
    a blank slot waiting to be filled with whatever text sits nearest.
    """
    from tests.conftest import MockLLMProvider

    eng = _engine(sample_persona, MockLLMProvider(["海边的照片"]), tmp_path,
                  provider="doubao")
    await eng._prepare_turn_v2("", [IMG], "feishu", "oc_test")

    assert "海边的照片" in eng.memory.short_term.get_history()[-1].content


@pytest.mark.asyncio
async def test_a_plain_text_turn_costs_no_describe_call(sample_persona, tmp_path):
    from tests.conftest import MockLLMProvider

    llm = MockLLMProvider(["unused"])
    eng = _engine(sample_persona, llm, tmp_path)
    await eng._prepare_turn_v2("在吗", None, "feishu", "oc_test")

    assert llm._call_count == 0
