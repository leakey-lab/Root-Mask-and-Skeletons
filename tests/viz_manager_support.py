"""Stand-ins of the dashboard-manager tests (``tests/test_viz_embed_manager*.py``).

A light stand-in for MainWindow (a real right_panel QStackedWidget and library QTreeWidget,
recording fakes for the rest) and stub dashboards that keep the real widgets' contract:
``cleanup_server()`` emits ``server_closed`` once while a server runs, plus ``export_saved``,
``export_failed`` and ``failed``. No browser and no Dash server start. Holds no assert.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtWidgets import (
    QMainWindow,
    QPushButton,
    QStackedWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QWidget,
)

from app.gui import visualization_manager as vm
from app.gui.panels import Panel

_PANELS = {panel.key: int(panel) for panel in Panel}


class StubDashboard(QMainWindow):
    """Keeps the real dashboards' contract without a browser or a server."""

    server_closed = pyqtSignal()
    export_saved = pyqtSignal(str)
    export_failed = pyqtSignal(str)
    made: list = []

    def __init__(self, csv_path):
        super().__init__()
        self.csv_path = csv_path
        self.failed = False
        self.cleanups = 0
        self._serving = True
        type(self).made.append(self)

    def cleanup_server(self):
        self.cleanups += 1
        if self._serving:  # emitted once, while a server runs
            self._serving = False
            self.server_closed.emit()


class StubLength(StubDashboard):
    made: list = []

    def __init__(self, csv_path, image_manager):
        super().__init__(csv_path)
        self.image_manager = image_manager


class StubArea(StubDashboard):
    made: list = []

    def __init__(self, csv_path, image_manager):
        super().__init__(csv_path)
        self.image_manager = image_manager


class FakeImageManager:
    def __init__(self, folder):
        self.original_folder = str(folder)

    def get_image_path(self, name):
        return os.path.join(self.original_folder, f"{name}.jpg")


class FakeSkeletonHandler:
    def __init__(self):
        self.busy_kinds: set[str] = set()
        self.asked: list[str] = []

    def is_busy(self, kind: str) -> bool:
        self.asked.append(kind)
        return kind in self.busy_kinds


class FakeMainWindow(QWidget):
    """The parts of MainWindow that visualization_manager uses."""

    def __init__(self, project, *, opengl: bool):
        super().__init__()
        self.right_panel = QStackedWidget(self)
        for name in _PANELS:
            page = QWidget()
            page.setObjectName(name)
            self.right_panel.addWidget(page)
        self.file_list = QTreeWidget(self)
        folder = QTreeWidgetItem(["T001"])
        leaf = QTreeWidgetItem(["img_001.jpg"])
        leaf.setData(0, Qt.ItemDataRole.UserRole, "img_001")
        folder.addChild(leaf)
        self.file_list.addTopLevelItem(folder)
        self.file_list.setCurrentItem(leaf)
        self.image_manager = FakeImageManager(project)
        self.skeleton_handler = FakeSkeletonHandler()
        self.visualize_root_length_button = QPushButton("Visualize Root Length", self)
        self.visualize_root_area_button = QPushButton("Visualize Root Area", self)
        self.root_length_viz = None
        self.root_area_viz = None
        self.leave_allowed = True  # False: an editor with unsaved edits refuses to close
        self.gl_calls: list[bool] = []
        self.refreshed: list = []
        self.notices: list[tuple[str, str]] = []
        # An OpenGL canvas viewport, as the display/editor views have by default.
        self._gl_viewport = QOpenGLWidget(self) if opengl else None

    def switch_right_panel(self, panel):
        if not self.leave_allowed:
            return False
        self.right_panel.setCurrentIndex(_PANELS[panel])
        return True

    def set_opengl_viewports_enabled(self, enabled):
        self.gl_calls.append(enabled)
        if enabled and self._gl_viewport is None:
            self._gl_viewport = QOpenGLWidget(self)
        elif not enabled and self._gl_viewport is not None:
            self._gl_viewport.setParent(None)
            self._gl_viewport.deleteLater()
            self._gl_viewport = None

    def on_image_selected(self, item):
        self.refreshed.append(item)

    def notify(self, message, kind="success", timeout=3200):
        self.notices.append((message, kind))

    # MainWindow's server_closed slots, verbatim.
    def on_visualization_server_closed(self):
        if self.root_length_viz:
            vm.close_root_length_visualization(self)

    def on_area_visualization_server_closed(self):
        if self.root_area_viz:
            vm.close_root_area_visualization(self)


def _environment(qapp, monkeypatch, project, *, opengl: bool):
    StubLength.made = []
    StubArea.made = []
    monkeypatch.setattr(vm, "RootLengthVisualization", StubLength)
    monkeypatch.setattr(vm, "RootAreaVisualization", StubArea)
    boxes: list[tuple[str, str, str]] = []

    class Box:
        @staticmethod
        def information(_parent, title, text, *_rest):
            boxes.append(("information", title, text))

        @staticmethod
        def warning(_parent, title, text, *_rest):
            boxes.append(("warning", title, text))

        @staticmethod
        def critical(_parent, title, text, *_rest):
            boxes.append(("critical", title, text))

    monkeypatch.setattr(vm, "QMessageBox", Box)
    return SimpleNamespace(mw=FakeMainWindow(project, opengl=opengl), boxes=boxes, project=project)


def _touch_newer(path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
