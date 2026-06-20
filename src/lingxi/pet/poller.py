"""Polls /pet/state and emits state_changed when the response differs.

Uses QTimer so callbacks happen on the GUI thread — no thread juggling
needed for window updates. Polling interval is ~3s by default which is
plenty: emotion/activity change at human-talking speed, not faster.

Network errors are swallowed silently. Pet stays on the last sprite, the
agent may just be restarting. Surfacing failures to the user would mean
showing tooltips/badges that distract from the actual ambient role.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from PyQt6.QtCore import QObject, QTimer, pyqtSignal


class StatePoller(QObject):
    """Polls /pet/state on a timer, emits state_changed on diff."""

    state_changed = pyqtSignal(dict)

    def __init__(
        self,
        url: str,
        interval_ms: int = 3000,
        timeout_s: float = 2.0,
    ):
        super().__init__()
        self.url = url
        self.timeout_s = timeout_s
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(interval_ms)
        self._last: dict | None = None

    def start(self) -> None:
        # Fire one immediately so initial sprite isn't idle_default for 3s
        self._tick()

    def stop(self) -> None:
        self._timer.stop()

    def _tick(self) -> None:
        try:
            with urllib.request.urlopen(self.url, timeout=self.timeout_s) as r:
                data = json.loads(r.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError, ValueError):
            return

        if not isinstance(data, dict):
            return

        # Emit when the sprite changes OR a new speech line arrives (the pet
        # can speak without changing pose, e.g. a comment while you keep
        # working — both want a redraw).
        new_sprite = data.get("sprite")
        old_sprite = self._last.get("sprite") if self._last else None
        new_seq = data.get("speech_seq", 0)
        old_seq = self._last.get("speech_seq", 0) if self._last else 0
        self._last = data
        if new_sprite != old_sprite or new_seq != old_seq:
            self.state_changed.emit(data)
