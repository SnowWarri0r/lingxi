"""Transparent, always-on-top, draggable sprite window.

The window itself has no chrome — just a QLabel holding the current PNG.
Frameless + WA_TranslucentBackground means the PNG's alpha defines the
visible shape, so the surrounding rectangle is fully click-through (clicks
on transparent pixels fall through to the desktop / app underneath via
the OS — no special handling needed on macOS for transparent regions of
a layered window).
"""

from __future__ import annotations

import json
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QGuiApplication, QPixmap
from PyQt6.QtWidgets import QApplication, QLabel, QMenu, QWidget


class PetWindow(QWidget):
    """Frameless transparent window that displays a sprite."""

    def __init__(
        self,
        sprite_dir: Path,
        pos_file: Path,
        size: tuple[int, int] = (200, 300),
    ):
        super().__init__()
        self.sprite_dir = sprite_dir
        self.pos_file = pos_file
        self._drag_offset = None
        self._current_sprite: str | None = None

        # Frameless + transparent + always on top.
        # Note: NO Tool flag on macOS — it conflicts with WindowStaysOnTopHint
        # under Frameless, causing the window to drop behind other windows.
        # Trade-off: pet shows in cmd-tab / dock, but stays reliably on top.
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        # Bypass any app-level blur/effects that some macOS WMs apply
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)

        self.resize(*size)

        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.resize(*size)
        # Important: label background must be transparent so the parent's
        # translucency shows through the empty alpha regions.
        self.label.setStyleSheet("background: transparent;")

        self._restore_pos()
        # Default sprite — only shows if file exists, otherwise window
        # appears empty (intentional — we don't want a "missing asset"
        # placeholder cluttering the desktop).
        self.show_sprite("idle_default")

    def show_sprite(self, name: str) -> None:
        if name == self._current_sprite:
            return
        path = self.sprite_dir / f"{name}.png"
        if not path.exists():
            return
        pix = QPixmap(str(path))
        if pix.isNull():
            return
        scaled = pix.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.label.setPixmap(scaled)
        self._current_sprite = name

    # --- drag to move ------------------------------------------------------

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_offset = (
                event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            )

    def mouseMoveEvent(self, event):
        if self._drag_offset and (event.buttons() & Qt.MouseButton.LeftButton):
            self.move(event.globalPosition().toPoint() - self._drag_offset)

    def mouseReleaseEvent(self, event):
        if self._drag_offset is not None:
            self._drag_offset = None
            self._save_pos()

    # --- right-click menu --------------------------------------------------

    def contextMenuEvent(self, event):
        menu = QMenu(self)
        hide = QAction("隐藏", self)
        hide.triggered.connect(self.hide)
        quit_a = QAction("退出", self)
        quit_a.triggered.connect(QApplication.instance().quit)
        menu.addAction(hide)
        menu.addSeparator()
        menu.addAction(quit_a)
        menu.exec(event.globalPos())

    # --- position persist --------------------------------------------------

    def _restore_pos(self) -> None:
        if self.pos_file.exists():
            try:
                data = json.loads(self.pos_file.read_text())
                self.move(int(data["x"]), int(data["y"]))
                return
            except (ValueError, OSError, KeyError, TypeError):
                pass
        # Fallback: bottom-right of primary screen, 40px margin
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
