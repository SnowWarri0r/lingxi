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

    app = QApplication(sys.argv)
    # Keep alive when window is hidden via right-click menu
    app.setQuitOnLastWindowClosed(False)

    window = PetWindow(sprite_dir=sprite_dir, pos_file=pos_file)
    window.show()

    poller = StatePoller(url=state_url)
    poller.state_changed.connect(
        lambda data: window.show_sprite(data.get("sprite", "idle_default"))
    )
    poller.start()

    print(f"[pet] running. sprites={sprite_dir} state={state_url}")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
