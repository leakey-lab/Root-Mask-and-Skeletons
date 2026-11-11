"""Helpers and fixtures of the Correct page's class-label tests (the split of the old
``test_correct_classes_editor.py``): the page with the class labels edited, its recorded
signals and notices, the prompts and the Remove GT box answered by the test, the class panel's
boxes, the folder listing compared by mtime and size.

Fixtures are imported by name into each test module (``from tests.correct_classes_support
import page as page  # fixture``). ``box`` and ``highlighted`` read the class panel's private
rows: ``ClassPanel`` has no public accessor for the box of a class or the highlighted row yet
(``box(key)``, ``highlighted_keys()`` are recorded for the owner); this is the one place to
change the day they exist.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QImage, QKeySequence
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QWidget,
)

import app.gt as gt_package
from app.gt import (
    Registry,
    Tracing,
    default_registry,
    registry_path,
    write_tracing,
)
from app.gui import editor_prompts
from app.gui.correct.editor import CorrectEditor
from app.gui.correct.layer import CLASSES
from app.gui.editor_prompts import CANCEL
from app.project_layout import ProjectLayout
from tests.correct_support import (
    NO_MODS,
    H,
    W,
    activate,
    drag,
    editor_factory,
    new_project,
    record,
    settle,
)
from tests.qt_teardown import delete_widgets
from tests.window_support import build_window, open_folder, quiet_window

ROOT = Path(__file__).resolve().parents[1]
APP_QSS = ROOT / "resources" / "themes" / "dark_theme.qss"
KEYS = ("root", "nodule", "root_tip")  # the default registry's labelled classes
Focus = Qt.FocusPolicy
REMOVE_DETAIL = (
    "Deletes gt/mask/a.png, gt/skeletons/a.png and gt/a.png where they exist. The predictions"
    " in mask/ and skeletons/ are kept. This cannot be undone."
)
PAGE_SIGNALS = ("gt_saved", "gt_removed", "mask_saved", "mask_cleared", "skeleton_cleared")


# ============================================================================== helpers
def norm(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))


def tree(folder: str) -> dict[str, tuple[int, int]]:
    """Every file under ``folder``: normalised path -> (mtime_ns, size)."""
    files = {}
    for root, _dirs, names in os.walk(folder):
        for name in names:
            path = os.path.join(root, name)
            stat = os.stat(path)
            files[norm(path)] = (stat.st_mtime_ns, stat.st_size)
    return files


def changed(before: dict, after: dict) -> set[str]:
    """The files written, made or removed between two :func:`tree` listings."""
    return {path for path in before.keys() | after.keys() if before.get(path) != after.get(path)}


def press(editor, key: str, modifiers=NO_MODS) -> None:
    """A key click on the canvas, as the user types it (``"G"``, ``"2"``)."""
    QTest.keyClick(editor.graphics_view, QKeySequence(key)[0].key(), modifiers)


def tap_b(editor) -> None:
    """B pressed and released on the canvas without a wheel turn."""
    QTest.keyClick(editor.graphics_view, Qt.Key.Key_B)


def paint(editor, *points) -> None:
    """A stroke of the edited layer's current tool (a drag on the canvas input)."""
    drag(editor, *points)


def ids(editor) -> np.ndarray:
    """The class-id plane of the open image."""
    return editor.layers[CLASSES].document.history.plane


def ticked(editor) -> list[str]:
    """The ticked boxes of the page's class panel."""
    return editor.class_panel.ticked_keys()


def highlighted(editor) -> list[str]:
    """The highlighted rows of the page's class panel."""
    return [key for key, row in editor.class_panel._rows.items() if row._active]


def assert_one_tick(editor, key: str) -> None:
    """Owner decision 2026-11-01: exactly one box is ticked, on the highlighted row, and it is
    the class the brush paints (the ring's colour too)."""
    layer = editor.layers[CLASSES]
    assert ticked(editor) == [key] == highlighted(editor), (ticked(editor), highlighted(editor))
    assert layer.active_class() == key
    assert layer.brush_color == QColor(layer.registry().by_key(key).color)


def box(editor, key: str) -> QCheckBox:
    """The box of class ``key`` on the page's class panel (ticked: the active class)."""
    widget = editor.class_panel._rows[key].checkbox
    assert isinstance(widget, QCheckBox)
    return widget


def row_text(editor, layer_id: str) -> str:
    return editor.layer_panel.rows[layer_id].state_label.text()


def walk(editor, key=Qt.Key.Key_Tab, presses: int = 80) -> list[QWidget]:
    """The widgets Tab (or Shift+Tab) visits from the canvas, until it is back there."""
    editor.graphics_view.setFocus(Qt.FocusReason.TabFocusReason)
    seen = []
    for _ in range(presses):
        QTest.keyClick(QApplication.focusWidget(), key)
        widget = QApplication.focusWidget()
        if widget is editor.graphics_view:
            return seen
        seen.append(widget)
    raise AssertionError(f"Tab never came back to the canvas: {seen}")


def pixels(widget) -> np.ndarray:
    QApplication.processEvents()
    image = widget.grab().toImage().convertToFormat(QImage.Format.Format_RGBA8888)
    rows = np.frombuffer(image.constBits().asstring(image.sizeInBytes()), np.uint8)
    width = image.width()
    return rows.reshape(image.height(), image.bytesPerLine())[:, : width * 4].reshape(
        image.height(), width, 4
    )


def write_unknown_class(image: str) -> bytes:
    """A labels file from another project: id 4 = 'hypha', unknown here (becomes Ignore)."""
    other = default_registry().add(key="hypha", name="Hypha", color="#3fa7ff")
    plane = np.zeros((H, W), np.uint8)
    plane[5:15, 5:15] = 4
    plane[30:40, 30:40] = 1
    path = ProjectLayout.for_image(image).gt_path("a")
    write_tracing(path, Tracing(plane), other)
    with open(path, "rb") as fh:
        return fh.read()


def fake_class_editor(monkeypatch, result: Registry | None, seen: list | None = None) -> None:
    """Replace the class editor with a stub that accepts with ``result``."""

    class _FakeDialog:
        def __init__(self, registry, parent=None):
            if seen is not None:
                seen.append(registry)

        def exec(self):
            return 1

        def result_registry(self):
            return result

        def deleteLater(self):  # noqa: N802 - Qt API
            pass

        def testAttribute(self, _attribute):  # noqa: N802 - Qt API (run_modal asks)
            return False

    fake = types.ModuleType("app.gui.class_editor_dialog")
    fake.ClassEditorDialog = _FakeDialog
    monkeypatch.setitem(sys.modules, "app.gui.class_editor_dialog", fake)


class Prompts:
    """The unsaved-changes box, answered with ``answer``; ``calls`` records each question."""

    def __init__(self) -> None:
        self.answer = CANCEL
        self.calls: list[tuple[str, str, str]] = []

    def ask_unsaved(self, parent, noun, name):
        self.calls.append(("ask_unsaved", noun, name))
        return self.answer


class Confirm:
    """The Remove GT box, answered with ``answer``; ``calls`` records ``(name, detail)``."""

    def __init__(self) -> None:
        self.answer = False
        self.calls: list[tuple[str, str]] = []

    def __call__(self, parent, name, detail) -> bool:
        self.calls.append((name, detail))
        return self.answer


# ============================================================================= fixtures


@pytest.fixture
def signals() -> dict[str, list]:
    """The page signals of the last page made, recorded (filled by ``make_editor``)."""
    return {}


@pytest.fixture
def notices() -> list[tuple[str, str]]:
    """Every notice of the last page made, from its build on (filled by ``make_editor``)."""
    return []


@pytest.fixture
def make_editor(app, signals, notices):
    """``make(image=None, base=None, *, size=(1100, 800), show=True)``: a CorrectEditor whose
    signals and notices are recorded before it opens ``image``."""
    factory = editor_factory()
    build = next(factory)

    def make(image=None, base=None, *, size=(1100, 800), show=True) -> CorrectEditor:
        editor = build(size=size, show=show)
        notices.clear()
        editor.notice.connect(lambda message, kind: notices.append((message, kind)))
        for name in PAGE_SIGNALS:
            signals[name] = record(getattr(editor, name))
        if image is not None:
            editor.load_image(image, base)
            settle(editor)
        return editor

    yield make
    for _ in factory:  # its teardown deletes every page made
        pass


@pytest.fixture
def project_lossy(project):
    """``project`` with a ``gt/a.png`` that holds a class unknown here (a lossy remap)."""
    write_unknown_class(project.a)
    return project


@pytest.fixture
def project_bad_registry(project):
    """``project`` with an invalid ``gt/classes.json``."""
    os.makedirs(os.path.join(project.folder, "gt"), exist_ok=True)
    with open(registry_path(project.folder), "w", encoding="utf-8") as fh:
        fh.write("{")
    return project


@pytest.fixture
def layout(project) -> ProjectLayout:
    return ProjectLayout.for_image(project.a)


@pytest.fixture
def prompts(monkeypatch) -> Prompts:
    fake = Prompts()
    monkeypatch.setattr(editor_prompts, "ask_unsaved", fake.ask_unsaved)
    return fake


@pytest.fixture
def page(make_editor, project):
    """The page on image a, the class labels edited, active, the keys on its canvas."""
    editor = make_editor(project.a, project.folder)
    editor.set_active_layer(CLASSES)
    activate(editor)
    return editor


@pytest.fixture
def themed(app):
    """The app stylesheet: a focus ring under test is the theme's."""
    previous = app.styleSheet()
    app.setStyleSheet(APP_QSS.read_text(encoding="utf-8"))
    yield app
    app.setStyleSheet(previous)


@pytest.fixture
def reads(monkeypatch) -> list[str]:
    """The labels files read (``gt/<stem>.png``)."""
    calls: list[str] = []
    real = gt_package.read_tracing

    def spy(path):
        calls.append(path)
        return real(path)

    monkeypatch.setattr(gt_package, "read_tracing", spy)
    return calls


@pytest.fixture
def made(app):
    """Widgets closed and deleted deterministically after the test."""
    widgets: list = []
    yield widgets
    delete_widgets(widgets)


# ------------------------------------------------------------------- the main window
@pytest.fixture
def confirm() -> Confirm:
    return Confirm()


@pytest.fixture
def window_with_project(app, tmp_path, no_temp_sweep, confirm):
    """A shown MainWindow on ``project`` with image a open in the Correct stage; the
    Remove GT box is ``confirm``."""
    project = new_project(tmp_path)
    window = build_window(tmp_path / "settings.ini")
    made = [window]
    open_folder(window, project.folder, 2)
    window.file_list.expandAll()
    item = window.library_index.item_for("a")
    window.file_list.setCurrentItem(item)
    window.on_tree_item_clicked(item, 0)
    settle()
    window.ribbon_buttons["Correct"].click()
    settle()
    window.correct_editor.actions.confirm_remove = confirm
    yield SimpleNamespace(window=window, project=project, editor=window.correct_editor)
    delete_widgets(made, prepare=quiet_window)
    QApplication.processEvents()
