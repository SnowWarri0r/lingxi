"""Entry point for `lingxi-pet`. Wires QApplication + window + poller."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _resolve_sprite_dir() -> Path:
    """Find the sprite directory.

    Priority:
      1. LINGXI_PET_SPRITES env var
      2. ./assets/pet/sprites (project root, when running via `uv run`)
      3. Package-relative fallback (when installed)
    """
    env = os.environ.get("LINGXI_PET_SPRITES", "").strip()
    if env:
        return Path(env).expanduser().resolve()

    cwd_candidate = Path.cwd() / "assets" / "pet" / "sprites"
    if cwd_candidate.exists():
        return cwd_candidate

    # Walk up from this file looking for assets/pet/sprites
    here = Path(__file__).resolve()
    for parent in here.parents:
        cand = parent / "assets" / "pet" / "sprites"
        if cand.exists():
            return cand
    return cwd_candidate  # return non-existing for clear error


def main() -> None:
    try:
        from PyQt6.QtWidgets import QApplication
    except ImportError:
        print("[pet] PyQt6 not installed. Install with:")
        print("      uv sync --extra pet")
        print("  or  uv pip install PyQt6")
        sys.exit(1)

    from lingxi.pet.poller import StatePoller
    from lingxi.pet.window import PetWindow

    sprite_dir = _resolve_sprite_dir()
    if not sprite_dir.exists():
        print(f"[pet] sprite directory not found: {sprite_dir}")
        print("[pet] set LINGXI_PET_SPRITES env var or run from project root")
        sys.exit(1)

    state_url = os.environ.get(
        "LINGXI_PET_STATE_URL", "http://127.0.0.1:7891/pet/state"
    )
    pos_file = Path.home() / ".lingxi" / "pet_pos.json"

    live2d = os.environ.get("LINGXI_PET_LIVE2D", "").strip() not in ("", "0", "false")
    if live2d:
        # QtWebEngine requires this attribute set BEFORE the QApplication is
        # constructed (otherwise: "must be imported ... before a QCoreApplication").
        from PyQt6.QtCore import Qt as _Qt
        QApplication.setAttribute(_Qt.ApplicationAttribute.AA_ShareOpenGLContexts)

    app = QApplication(sys.argv)
    # Keep alive when window is hidden via right-click menu
    app.setQuitOnLastWindowClosed(False)

    # Live2D mode: a rigged, genuinely-animated body (breathing/blink/sway/
    # eye-tracking) instead of the flat sprite. Opt-in via env while it's V1.
    if live2d:
        from lingxi.pet.live2d_window import Live2DWindow
        from lingxi.pet.poller import StatePoller

        html_path = (
            Path(os.environ.get("LINGXI_PET_LIVE2D_HTML", "").strip())
            if os.environ.get("LINGXI_PET_LIVE2D_HTML", "").strip()
            else sprite_dir.parent / "live2d" / "index.html"
        )
        if not html_path.exists():
            print(f"[pet] live2d html not found: {html_path}")
            sys.exit(1)

        pos_file = Path.home() / ".lingxi" / "pet_pos.json"
        # LINGXI_PET_MODEL = any Live2D model3.json URL (free sample / bought /
        # commissioned 妮妮). Empty → the page's built-in default.
        model_url = os.environ.get("LINGXI_PET_MODEL", "").strip()
        window = Live2DWindow(html_path, pos_file, model_url=model_url)
        window.show()

        poller = StatePoller(url=state_url)
        _last = {"v": 0}

        def on_state(data: dict) -> None:
            seq = data.get("speech_seq", 0)
            text = data.get("speech", "")
            if seq and seq != _last["v"] and text:
                _last["v"] = seq
                window.say(text)

        poller.state_changed.connect(on_state)
        poller.start()
        print(f"[pet] live2d running. html={html_path} state={state_url}")
        sys.exit(app.exec())

    window = PetWindow(sprite_dir=sprite_dir, pos_file=pos_file)
    window.show()
    # On macOS, raise_() after show() is what actually pulls the window
    # to the top of its level. Without this, StaysOnTopHint is "best effort"
    # and the OS sometimes parks it underneath the focused app.
    window.raise_()
    window.activateWindow()

    poller = StatePoller(url=state_url)

    _last_speech_seq = {"v": 0}

    def on_state(data: dict) -> None:
        window.show_sprite(data.get("sprite", "idle_default"))
        seq = data.get("speech_seq", 0)
        text = data.get("speech", "")
        if seq and seq != _last_speech_seq["v"] and text:
            _last_speech_seq["v"] = seq
            window.show_speech(text)

    poller.state_changed.connect(on_state)
    poller.start()

    print(f"[pet] running. sprites={sprite_dir} state={state_url}")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
