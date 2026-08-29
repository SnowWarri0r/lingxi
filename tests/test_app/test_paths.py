"""Where each kind of state lives."""

import os
from pathlib import Path

from lingxi.paths import persona_data_root, stickers_root


class _P:
    slug = "tangkeke"


def test_persona_state_is_namespaced_per_persona():
    assert persona_data_root(_P()) == os.path.join("data", "personas", "tangkeke")


def test_stickers_are_shared_and_not_under_the_persona_namespace():
    """Stickers are captioned reaction images with nothing persona-specific.

    This was written as `data_dir.parent / "stickers"`, which meant
    data/stickers/ while data_dir was data/memory — and silently became
    data/personas/stickers/ when per-persona namespacing moved data_dir.
    That directory got created empty on startup, every lookup returned
    nothing, and she stopped sending stickers with no error anywhere.
    """
    root = Path(stickers_root())
    assert root == Path("data") / "stickers"
    assert "personas" not in root.parts
    # And specifically not derivable from a persona's dir any more.
    assert root != Path(persona_data_root(_P())).parent / "stickers"


def test_stickers_root_is_overridable(monkeypatch):
    monkeypatch.setenv("LINGXI_STICKERS_DIR", "/tmp/pool")
    assert stickers_root() == "/tmp/pool"
