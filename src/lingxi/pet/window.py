"""Transparent, always-on-top sprite window with life signs.

clawd-on-desk-inspired features layered on the original PyQt6 base:

- **APNG / GIF animated sprites** when an animated file exists for a state
  (e.g. `eating.apng`); falls back to the static `eating.png`. Lets us
  iteratively replace specific states with animation without breaking
  others.
- **Idle breathing** for static-PNG idle states — subtle 3% scale cycle
  at ~3s per breath. Costs nothing extra (just QTimer + 4 cached scales)
  and immediately makes the pet read as "alive" instead of "decal stuck
  on the screen".
- **Edge hiding mini mode** — drag to a screen edge, pet snaps and hides
  ~70% offscreen. Hover the visible sliver and it pops back out. Lets
  the pet stay on-call without consuming attention.

Click-through behavior on transparent PNG pixels remains as before: the
window itself has no chrome; alpha defines the interactive shape.
"""

from __future__ import annotations

import json
from pathlib import Path

from PyQt6.QtCore import QPoint, Qt, QTimer
from PyQt6.QtGui import QAction, QGuiApplication, QMovie, QPixmap
from PyQt6.QtWidgets import QApplication, QLabel, QMenu, QWidget


# States that get the breathing animation when shown as static PNG. For
# active states (eating / chatting) breathing reads as fidgety, so skip.
_IDLE_STATES = frozenset({
    "idle_default",
    "happy",
    "tired",
    "sleepy",
    "withdrawn",
    "focused",
})

# How close to a screen edge before snapping (px, in logical pixels).
_EDGE_SNAP_PX = 24

# Fraction of the pet that's hidden offscreen when snapped to an edge.
_HIDE_FRACTION = 0.65


class PetWindow(QWidget):
    """Frameless transparent window that displays a (possibly animated) sprite."""

    def __init__(
        self,
        sprite_dir: Path,
        pos_file: Path,
        size: tuple[int, int] = (240, 270),
    ):
        super().__init__()
        self.sprite_dir = sprite_dir
        self.pos_file = pos_file
        self._drag_offset = None
        self._dragging = False
        self._current_sprite: str | None = None
        self._movie: QMovie | None = None  # active QMovie when animated

        # Edge-hide state: None when pet is in normal floating mode; tuple
        # ('left'|'right'|'top', hidden_pos, full_pos) when snapped.
        self._snap: tuple[str, QPoint, QPoint] | None = None

        # Frameless + transparent + always on top.
        # No Tool flag on macOS — it conflicts with WindowStaysOnTopHint
        # under Frameless, causing the window to drop behind other windows.
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)

        self.resize(*size)

        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setScaledContents(False)
        self.label.resize(*size)
        self.label.setStyleSheet("background: transparent;")

        # Track mouse moves over the label so hover-peek works even though
        # the widget is large but mostly transparent. Qt only delivers
        # mouseMove with no buttons pressed when setMouseTracking is on.
        self.setMouseTracking(True)
        self.label.setMouseTracking(True)

        # Breathing animation state (used only for static idle sprites)
        self._breath_timer = QTimer(self)
        self._breath_timer.timeout.connect(self._tick_breath)
        self._breath_frames: list[QPixmap] = []
        self._breath_phase = 0

        # Peek timer — slides back to hidden after mouse leaves the body
        self._unpeek_timer = QTimer(self)
        self._unpeek_timer.setSingleShot(True)
        self._unpeek_timer.timeout.connect(self._slide_to_hidden)

        self._restore_pos()
        self.show_sprite("idle_default")

    # --- sprite display ----------------------------------------------------

    def show_sprite(self, name: str) -> None:
        if name == self._current_sprite:
            return

        # Try animated formats first; fall back to static PNG.
        for ext in ("apng", "gif"):
            anim_path = self.sprite_dir / f"{name}.{ext}"
            if anim_path.exists() and self._show_animated(anim_path):
                self._current_sprite = name
                self._stop_breathing()
                return

        path = self.sprite_dir / f"{name}.png"
        if not path.exists():
            return
        pix = QPixmap(str(path))
        if pix.isNull():
            return

        self._stop_movie()
        scaled = self._scale_for_display(pix)
        self.label.setPixmap(scaled)
        self._current_sprite = name

        # Idle states get a subtle breathing animation; active states stay
        # still so eating/chatting don't read as fidgety.
        if name in _IDLE_STATES:
            self._start_breathing(pix)
        else:
            self._stop_breathing()

    def _show_animated(self, path: Path) -> bool:
        """Display an APNG/GIF via QMovie. Returns True on success."""
        movie = QMovie(str(path))
        if not movie.isValid():
            return False
        self._stop_movie()
        # Native pixel rendering at the size we want
        ratio = self.devicePixelRatioF()
        target = self.size() * ratio
        movie.setScaledSize(target)
        self.label.setMovie(movie)
        movie.start()
        self._movie = movie
        return True

    def _stop_movie(self) -> None:
        if self._movie is not None:
            self._movie.stop()
            self.label.setMovie(None)
            self._movie = None

    def _scale_for_display(self, pix: QPixmap, scale: float = 1.0) -> QPixmap:
        """Scale a pixmap to the window size at native pixel resolution."""
        ratio = self.devicePixelRatioF()
        target = self.size() * ratio * scale
        scaled = pix.scaled(
            target,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        scaled.setDevicePixelRatio(ratio)
        return scaled

    # --- breathing animation ----------------------------------------------

    def _start_breathing(self, base_pix: QPixmap) -> None:
        """Pre-render a few subtle scale variants and cycle through them.

        4 frames at 800ms = 3.2s per breath cycle. Scale range ±1.5% is
        barely perceptible per frame but reads as a clear sign of life
        across the cycle. No new assets needed.
        """
        scales = (1.000, 1.010, 1.020, 1.010)
        self._breath_frames = [
            self._scale_for_display(base_pix, scale=s) for s in scales
        ]
        self._breath_phase = 0
        self._breath_timer.start(800)

    def _stop_breathing(self) -> None:
        self._breath_timer.stop()
        self._breath_frames = []

    def _tick_breath(self) -> None:
        if not self._breath_frames:
            return
        self._breath_phase = (self._breath_phase + 1) % len(self._breath_frames)
        self.label.setPixmap(self._breath_frames[self._breath_phase])

    # --- drag to move ------------------------------------------------------

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_offset = (
                event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            )
            self._dragging = False  # becomes True after first real move

    def mouseMoveEvent(self, event):
        # Drag mode — left button held
        if self._drag_offset and (event.buttons() & Qt.MouseButton.LeftButton):
            self._dragging = True
            # While the user drags, cancel any pending hide and treat the
            # pet as un-snapped — they're repositioning.
            self._unpeek_timer.stop()
            self._snap = None
            self.move(event.globalPosition().toPoint() - self._drag_offset)
            return

        # Hover mode — peek when mouse touches body of a snapped pet
        if self._snap is not None:
            self._slide_to_full()
            self._unpeek_timer.stop()

    def mouseReleaseEvent(self, event):
        was_dragging = self._dragging
        self._dragging = False
        if self._drag_offset is not None:
            self._drag_offset = None
            if was_dragging:
                self._maybe_snap_to_edge()
            self._save_pos()

    def leaveEvent(self, event):
        # Mouse left the body — if snapped, slide back to hidden after a beat
        if self._snap is not None:
            self._unpeek_timer.start(600)
        super().leaveEvent(event)

    # --- edge snap / hide --------------------------------------------------

    def _maybe_snap_to_edge(self) -> None:
        """If the pet ended up near a screen edge, snap and partial-hide."""
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            self._snap = None
            return
        avail = screen.availableGeometry()
        x, y, w, h = self.x(), self.y(), self.width(), self.height()
        hidden_w = int(w * _HIDE_FRACTION)

        # Right edge
        if x + w >= avail.x() + avail.width() - _EDGE_SNAP_PX:
            hidden = QPoint(avail.x() + avail.width() - (w - hidden_w), y)
            full = QPoint(avail.x() + avail.width() - w, y)
            self._snap = ("right", hidden, full)
            self.move(hidden)
            return
        # Left edge
        if x <= avail.x() + _EDGE_SNAP_PX:
            hidden = QPoint(avail.x() - hidden_w, y)
            full = QPoint(avail.x(), y)
            self._snap = ("left", hidden, full)
            self.move(hidden)
            return
        # Top edge (less common — let it hide too)
        if y <= avail.y() + _EDGE_SNAP_PX:
            hidden_h = int(h * _HIDE_FRACTION)
            hidden = QPoint(x, avail.y() - hidden_h)
            full = QPoint(x, avail.y())
            self._snap = ("top", hidden, full)
            self.move(hidden)
            return

        self._snap = None

    def _slide_to_full(self) -> None:
        if self._snap is None:
            return
        _, _, full = self._snap
        self.move(full)

    def _slide_to_hidden(self) -> None:
        if self._snap is None:
            return
        _, hidden, _ = self._snap
        self.move(hidden)

    # --- right-click menu --------------------------------------------------

    def contextMenuEvent(self, event):
        menu = QMenu(self)
        if self._snap is not None:
            unsnap = QAction("从边缘拉回", self)
            unsnap.triggered.connect(self._unsnap)
            menu.addAction(unsnap)
            menu.addSeparator()
        hide = QAction("隐藏", self)
        hide.triggered.connect(self.hide)
        quit_a = QAction("退出", self)
        quit_a.triggered.connect(QApplication.instance().quit)
        menu.addAction(hide)
        menu.addSeparator()
        menu.addAction(quit_a)
        menu.exec(event.globalPos())

    def _unsnap(self) -> None:
        if self._snap is None:
            return
        _, _, full = self._snap
        self.move(full)
        self._snap = None
        self._save_pos()

    # --- position persist --------------------------------------------------

    def _restore_pos(self) -> None:
        if self.pos_file.exists():
            try:
                data = json.loads(self.pos_file.read_text())
                self.move(int(data["x"]), int(data["y"]))
                return
            except (ValueError, OSError, KeyError, TypeError):
                pass
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            return
        avail = screen.availableGeometry()
        self.move(
            avail.x() + avail.width() - self.width() - 40,
            avail.y() + avail.height() - self.height() - 40,
        )

    def _save_pos(self) -> None:
        try:
            self.pos_file.parent.mkdir(parents=True, exist_ok=True)
            self.pos_file.write_text(json.dumps({"x": self.x(), "y": self.y()}))
        except OSError:
            pass
