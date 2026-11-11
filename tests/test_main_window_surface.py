"""The main window's surface (``app/gui/main_window.py``, ``app/gui/window/``).

- Every name a module outside the window reads on it (``main_window.file_list``,
  ``mw.switch_right_panel``, ``self.main_window.load_results`` ...) exists on a built
  window. The names come from the callers (an AST scan of ``app/``), not from a frozen
  list: a rename inside the window that misses a caller fails here, also when the caller
  runs only in an error path.
- No call hop (``CLAUDE.md``): ``MainWindow`` is built from the parts in
  ``app/gui/window/``; their functions are bound as the window's methods, so a signal on a
  hot path (arrow-key browse, PgDn, a stage switch, a finished job) runs the part's
  function itself, with no forwarding method of the window in between.
"""

from __future__ import annotations

import ast
import os
import sys
import textwrap

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("ROOT_VIEWER_ENABLE_OPENGL_VIEWPORT", "0")

from PyQt6.QtCore import QSettings, QThreadPool  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from app.gui import main_window  # noqa: E402
from app.gui.panels import Panel  # noqa: E402
from app.gui.window import body, job_buttons, navigator  # noqa: E402
from tests.qt_teardown import delete_widgets  # noqa: E402
from tests.source_scan_support import app_files, parse, rel  # noqa: E402

#: The names the modules outside the window give it (a parameter, an attribute of self).
RECEIVERS = frozenset({"main_window", "_main_window", "mw", "_mw"})


def _receiver(node: ast.AST) -> str | None:
    """``main_window`` for ``main_window`` and for ``self.main_window``."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        return node.attr if node.value.id == "self" else None
    return None


def window_names_read(tree: ast.Module) -> set[str]:
    """The attributes ``tree`` reads (or sets, or calls) on a window receiver."""
    return {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and _receiver(node.value) in RECEIVERS
    }


def _outside_the_window() -> dict[str, set[str]]:
    """``{name: {file, ...}}`` over ``app/`` without the window's own code."""
    found: dict[str, set[str]] = {}
    for path in app_files():
        where = rel(path)
        if where == "app/gui/main_window.py" or where.startswith("app/gui/window/"):
            continue
        for name in window_names_read(parse(path)):
            found.setdefault(name, set()).add(where)
    return found


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def window(app, tmp_path_factory):
    settings = QSettings(
        str(tmp_path_factory.mktemp("surface") / "s.ini"), QSettings.Format.IniFormat
    )
    made = [main_window.MainWindow(settings=settings)]
    yield made[0]
    QThreadPool.globalInstance().waitForDone(10_000)
    delete_widgets(made)


APP_DIR = os.path.normcase(os.path.dirname(os.path.dirname(main_window.__file__)))


def _callers(action, function) -> list[str]:
    """Who called ``function`` while ``action`` ran: ``"Qt"`` when no code of ``app/`` sits
    between the signal (or the test) and it, else that caller's ``module:function``."""
    callers: list[str] = []

    def hook(frame, event, _arg):
        if event == "call" and frame.f_code is function.__code__:
            back = frame.f_back
            if back is None or not os.path.normcase(back.f_code.co_filename).startswith(APP_DIR):
                callers.append("Qt")
            else:
                callers.append(f"{back.f_globals.get('__name__')}:{back.f_code.co_name}")

    sys.setprofile(hook)
    try:
        action()
    finally:
        sys.setprofile(None)
    return callers


def test_the_hot_paths_reach_the_parts_without_a_forwarding_call(window):
    """Arrow-key browse (the tree's current item, the debounce), PgDn in an editor, a stage
    switch and a finished job: the signal runs the part's function itself, with no code of
    the app (such as a forwarding method of the window) in between."""
    tree = window.file_list
    emit = tree.currentItemChanged.emit
    assert _callers(lambda: emit(None, None), navigator.on_tree_current_changed) == ["Qt"]
    timer = window._tree_load_timer
    timer.setInterval(0)
    try:
        timer.start()

        def until_fired():
            for _ in range(100):
                QApplication.processEvents()
                if not timer.isActive():
                    return

        assert _callers(until_fired, navigator.load_current_tree_item) == ["Qt"]
    finally:
        timer.setInterval(main_window.TREE_LOAD_DELAY_MS)
    emit = window.correct_editor.navigate_requested.emit
    assert _callers(lambda: emit(1), navigator.navigate_image) == ["Qt"]
    emit = window.right_panel.currentChanged.emit
    assert _callers(lambda: emit(Panel.DISPLAY), body.sync_library_for_panel) == ["Qt"]
    assert _callers(lambda: emit(Panel.DISPLAY), navigator.show_selection_on_display) == ["Qt"]
    emit = window.jobs.busy_changed.emit
    assert _callers(lambda: emit(False), job_buttons.on_jobs_busy_changed) == ["Qt"]


def test_the_scan_finds_the_window_names_a_module_reads():
    source = textwrap.dedent(
        """
        def build(main_window):
            main_window.planted_button.setText("x")
            main_window.planted_slot()

        class Manager:
            def show(self):
                self._mw.planted_switch("display")
                self.main_window.planted_attribute = 1
                self.other.not_a_window_name
                window.not_a_receiver
        """
    )
    assert window_names_read(ast.parse(source)) == {
        "planted_button",
        "planted_slot",
        "planted_switch",
        "planted_attribute",
    }


def test_every_window_name_another_module_reads_exists(window):
    read = _outside_the_window()
    assert len(read) > 20, "the scan found almost no window names: it would pass vacuously"
    missing = {name: sorted(files) for name, files in read.items() if not hasattr(window, name)}
    assert missing == {}, "modules read these names on the main window, which has none of them"
