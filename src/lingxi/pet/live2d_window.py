"""Live2D pet window — a transparent QWebEngineView hosting a Live2D model.

Unlike the flat-sprite window, the character is rigged into deformable parts:
breathing, blinking, idle sway, and eyes/head following the cursor all come
from the Live2D runtime (pixi-live2d-display), not from scaling a PNG. This is
the "actually alive" body. Libs are vendored under assets/pet/live2d/vendor;
the placeholder model is fetched from CDN for V1 (replaced by the 妮妮 model
in V2).

QWebEngineView renders into an internal child that swallows mouse events, so
window drag + right-click menu go through an event filter on the focus proxy.
"""

from __future__ import annotations

import json
from pathlib import Path

from PyQt6.QtCore import QEvent, QPoint, Qt, QTimer, QUrl, QUrlQuery
from PyQt6.QtGui import QColor, QCursor
from PyQt6.QtWebEngineCore import QWebEngineSettings
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWidgets import QApplication, QMenu


class Live2DWindow(QWebEngineView):
    def __init__(
        self,
        html_path: Path,
        pos_file: Path,
        size: tuple[int, int] = (300, 380),
        model_url: str = "",
    ) -> None:
        super().__init__()
        self.pos_file = pos_file
        self._drag_offset: QPoint | None = None

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.page().setBackgroundColor(QColor(0, 0, 0, 0))
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
        self.resize(*size)

        # The local index.html (file://) fetches the placeholder model from a
        # CDN — allow that. (V2 vendors the model locally and this can drop.)
        s = self.settings()
        s.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
        s.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True)

        url = QUrl.fromLocalFile(str(html_path))
        if model_url:
            q = QUrlQuery()
            q.addQueryItem("model", model_url)
            url.setQuery(q)
        self.load(url)
        self._restore_pos()

        # Feed the global cursor position into the model so her eyes/head track
        # the mouse even while it's outside this transparent window.
        self._focus_timer = QTimer(self)
        self._focus_timer.timeout.connect(self._push_cursor)
        self._focus_timer.start(120)

    # ---- speech (called by app on a new companion line) ------------------
    def say(self, text: str) -> None:
        self.page().runJavaScript(f"window.petSay && petSay({json.dumps(text)})")

    # ---- cursor tracking -------------------------------------------------
    def _push_cursor(self) -> None:
        local = self.mapFromGlobal(QCursor.pos())
        x = min(max(local.x(), 0), self.width())
        y = min(max(local.y(), 0), self.height())
        self.page().runJavaScript(f"window.petFocus && petFocus({x},{y})")

    # ---- drag + menu via event filter on the web view's child ------------
    def showEvent(self, e) -> None:  # noqa: ANN001
        super().showEvent(e)
        fp = self.focusProxy()
        if fp is not None:
            fp.installEventFilter(self)

    def eventFilter(self, obj, ev) -> bool:  # noqa: ANN001
        t = ev.type()
        if t == QEvent.Type.MouseButtonPress:
            if ev.button() == Qt.MouseButton.LeftButton:
                self._drag_offset = (
                    ev.globalPosition().toPoint() - self.frameGeometry().topLeft()
                )
            elif ev.button() == Qt.MouseButton.RightButton:
                self._show_menu(ev.globalPosition().toPoint())
                return True
        elif t == QEvent.Type.MouseMove and self._drag_offset is not None:
            if ev.buttons() & Qt.MouseButton.LeftButton:
                self.move(ev.globalPosition().toPoint() - self._drag_offset)
        elif t == QEvent.Type.MouseButtonRelease:
            if self._drag_offset is not None:
                self._save_pos()
            self._drag_offset = None
        return super().eventFilter(obj, ev)

    def _show_menu(self, global_pos: QPoint) -> None:
        m = QMenu()
        m.addAction("隐藏", self.hide)
        m.addAction("退出", QApplication.quit)
        m.exec(global_pos)

    # ---- position persistence -------------------------------------------
    def _restore_pos(self) -> None:
        try:
            data = json.loads(self.pos_file.read_text())
            self.move(int(data["x"]), int(data["y"]))
        except Exception:
            screen = QApplication.primaryScreen().availableGeometry()
            self.move(screen.right() - self.width() - 40,
                      screen.bottom() - self.height() - 40)

    def _save_pos(self) -> None:
        try:
            self.pos_file.parent.mkdir(parents=True, exist_ok=True)
            self.pos_file.write_text(
                json.dumps({"x": self.x(), "y": self.y()})
            )
        except Exception:
            pass
