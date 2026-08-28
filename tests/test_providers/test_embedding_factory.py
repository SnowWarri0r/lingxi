"""The doubao embedding backend refuses to build without a model.

The default used to name a public model that has since been retired. Every
call came back `404 does not exist or you do not have access to it`, which
reads as a permissions problem and sends you looking at the wrong thing —
the real cause was a setting nobody had filled in. An ARK endpoint id belongs
to one account, so no default can ship in the repo; say what is missing.
"""

import pytest

from lingxi.providers.embedding import DoubaoEmbeddingProvider, create_embedding_provider


@pytest.fixture
def ark_key(monkeypatch):
    monkeypatch.setenv("ARK_API_KEY", "sk-test")


def test_no_model_means_no_provider(ark_key, capsys):
    assert create_embedding_provider(kind="doubao", model=None) is None


def test_it_names_the_setting_to_fill_in(ark_key, capsys):
    create_embedding_provider(kind="doubao", model=None)

    out = capsys.readouterr().out
    assert "EMBEDDING_MODEL" in out, "the message has to say what to set"


def test_an_empty_string_is_not_a_model(ark_key):
    assert create_embedding_provider(kind="doubao", model="") is None


def test_a_configured_endpoint_builds(ark_key):
    provider = create_embedding_provider(kind="doubao", model="ep-20260101abc")

    assert isinstance(provider, DoubaoEmbeddingProvider)
    assert provider._model == "ep-20260101abc"


def test_a_missing_key_still_reports_the_key(monkeypatch, capsys):
    monkeypatch.delenv("ARK_API_KEY", raising=False)
    assert create_embedding_provider(kind="doubao", model="ep-x") is None
    assert "ARK_API_KEY" in capsys.readouterr().out


def test_the_provider_cannot_be_built_without_naming_a_model():
    """Positional construction used to inherit the dead default silently."""
    with pytest.raises(TypeError):
        DoubaoEmbeddingProvider(api_key="sk-test")
