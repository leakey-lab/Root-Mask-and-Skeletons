"""The status line (``app/gui/shell_chrome.py`` ``StatusLine``): the window's one status
surface.

Promises: QStatusBar messages show in it and the bar itself is hidden; the task progress
runs inline; with no message it says ``Ready · N images`` (or what the stage on screen would
work on), ``Working…`` while a task runs, and re-reads that text when asked; the dot follows
the task's state and returns to idle when a failure row hides; the device chip never imports
torch, names the resolved device, warns when a model is missing, and keeps its full text in
the tooltip.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QMainWindow, QStatusBar

from app.gui import canvas_gl, task_progress
from app.gui.shell_chrome import WORKING_TEXT, StatusLine, _idle_text_for, runtime_label
from app.gui.task_progress import TaskProgressWidget
from tests.qt_teardown import delete_widgets
from tests.qt_wait import wait_until as _wait
from tests.window_support import build_window, write_photos

pytestmark = pytest.mark.usefixtures("no_temp_sweep")


RUNTIME = "app.inference.runtime"


CANVAS = "OpenGL · Fake GPU"


def _window(idle=None):
    win = QMainWindow()
    line = StatusLine(idle_text=idle)
    win.setCentralWidget(line)
    bar = QStatusBar()
    win.setStatusBar(bar)
    tp = TaskProgressWidget()
    bar.addPermanentWidget(tp)
    line.adopt(bar, tp)
    return win, line, bar, tp


def test_status_bar_messages_show_in_the_statusline(app):
    win, line, bar, _tp = _window()
    bar.showMessage("Calculating root lengths...")
    assert line.message.text() == "Calculating root lengths..."
    bar.clearMessage()
    assert line.message.text() == "Ready"


def test_idle_text_is_injectable(app):
    win, line, bar, _tp = _window(idle=lambda: "Ready · 40 images")
    bar.showMessage("x")
    bar.clearMessage()
    assert line.message.text() == "Ready · 40 images"


def test_adopt_hides_the_status_bar_and_hosts_progress(app):
    win, line, bar, tp = _window()
    win.show()
    app.processEvents()
    assert not bar.isVisible()
    assert tp.parentWidget() is line
    tp.start("Generating masks")
    app.processEvents()
    assert tp.isVisible()
    assert line.state() == "working"
    tp.fail("nope")
    assert line.state() == "error"
    win.close()


def test_adopt_is_idempotent(app):
    win, line, bar, tp = _window()
    line.adopt(bar, tp)  # MainWindow may adopt again: nothing is connected twice
    assert bar.receivers(bar.messageChanged) == 1
    assert tp.receivers(tp.state_changed) == 1
    bar.showMessage("once")
    assert line.message.text() == "once"


def _line(idle=None):
    win = QMainWindow()
    line = StatusLine(idle_text=idle)
    win.setCentralWidget(line)
    bar = QStatusBar()
    win.setStatusBar(bar)
    progress = TaskProgressWidget()
    bar.addPermanentWidget(progress)
    line.adopt(bar, progress)
    return win, line, bar, progress


def test_a_running_task_does_not_say_ready(app):
    _win, line, bar, progress = _line(idle=lambda: "Ready · 40 images")
    progress.start("Generating masks", cancelable=True)
    assert line.message.text() == WORKING_TEXT
    bar.showMessage("Loading the mask model…")
    assert line.message.text() == "Loading the mask model…"  # messages still win
    bar.clearMessage()
    assert line.message.text() == WORKING_TEXT
    progress.finish("Masks generated")
    assert line.message.text() == "Ready · 40 images"


def test_the_dot_returns_to_idle_when_a_failure_row_hides(app, monkeypatch):
    monkeypatch.setattr(task_progress, "FAIL_HIDE_MS", 40)
    _win, line, _bar, progress = _line()
    progress.start("Measuring")
    progress.fail("Measure failed: disk full")
    assert line.state() == "error"
    assert _wait(lambda: progress.isHidden())
    assert line.state() == "idle"


def test_a_new_task_before_the_failure_row_hides_keeps_working(app, monkeypatch):
    monkeypatch.setattr(task_progress, "FAIL_HIDE_MS", 40)
    _win, line, _bar, progress = _line()
    states = []
    progress.state_changed.connect(states.append)
    progress.start("first")
    progress.fail("nope")
    progress.start("second")
    QTest.qWait(150)
    QApplication.processEvents()
    assert states == ["working", "error", "working"]
    assert line.state() == "working"


def test_refresh_idle_rereads_the_provider_without_a_message(app):
    count = [40]
    _win, line, bar, _progress = _line(idle=lambda: f"Ready · {count[0]} images")
    count[0] = 768
    line.refresh_idle()
    assert line.message.text() == "Ready · 768 images"
    bar.showMessage("Saving…")
    count[0] = 5
    line.refresh_idle()
    assert line.message.text() == "Saving…"  # a message on show is kept


def _text(count: int) -> str:
    window = SimpleNamespace(image_manager=SimpleNamespace(images=dict.fromkeys(range(count))))
    return _idle_text_for(window)()


def test_one_image_is_singular():
    assert _text(1) == "Ready · 1 image"


def test_many_images_have_a_thousands_separator():
    assert _text(3) == "Ready · 3 images"
    assert _text(46178) == "Ready · 46,178 images"


def test_no_images_is_just_ready():
    assert _text(0) == "Ready"


def test_runtime_label_never_imports_the_runtime(monkeypatch):
    monkeypatch.delitem(sys.modules, RUNTIME, raising=False)
    assert runtime_label() == "Device: auto"
    assert RUNTIME not in sys.modules


def _fake_runtime(monkeypatch, *, resolved, cuda, name="NVIDIA RTX"):
    mod = types.ModuleType(RUNTIME)
    cfg = types.SimpleNamespace(is_cuda=cuda, device_name=name if cuda else "cpu")

    def get_runtime():
        return cfg

    get_runtime.cache_info = lambda: types.SimpleNamespace(currsize=1 if resolved else 0)
    mod.get_runtime = get_runtime
    monkeypatch.setitem(sys.modules, RUNTIME, mod)
    return mod


def test_runtime_label_cpu(monkeypatch):
    _fake_runtime(monkeypatch, resolved=True, cuda=False)
    assert runtime_label() == "CPU"


def test_runtime_label_gpu(monkeypatch):
    _fake_runtime(monkeypatch, resolved=True, cuda=True)
    assert runtime_label() == "GPU · NVIDIA RTX"


def test_runtime_label_unresolved_does_not_probe(monkeypatch):
    mod = _fake_runtime(monkeypatch, resolved=False, cuda=True)
    mod.get_runtime = types.SimpleNamespace(cache_info=mod.get_runtime.cache_info)  # not callable
    assert runtime_label() == "Device: auto"


def test_runtime_label_prefers_describe_runtime(monkeypatch):
    mod = _fake_runtime(monkeypatch, resolved=True, cuda=False)
    mod.describe_runtime = lambda: "CPU · masks ok"
    assert runtime_label() == "CPU · masks ok"


def test_runtime_summary_object_and_missing_models_warn(app, monkeypatch):
    monkeypatch.setattr(canvas_gl, "canvas_line", lambda: (CANVAS, False))
    mod = _fake_runtime(monkeypatch, resolved=True, cuda=False)
    mod.describe_runtime = lambda: types.SimpleNamespace(
        text="CPU · FP32 · mask model missing", models_ok=False
    )
    _win, line, _bar, _tp = _window()
    line.refresh_runtime()
    assert line.info.toolTip() == (
        f"Inference runtime: CPU · FP32 · mask model missing\nCanvas: {CANVAS}"
    )
    assert line.info.property("models_ok") is False


def test_long_runtime_text_is_elided(app, monkeypatch):
    mod = _fake_runtime(monkeypatch, resolved=True, cuda=False)
    mod.describe_runtime = lambda: "GPU · " + "X" * 400
    _win, line, _bar, _tp = _window()
    line.refresh_runtime()
    assert line.info.sizeHint().width() <= 340
    assert "X" * 400 in line.info.toolTip()


def test_statusline_chip_is_not_hard_coded(app, monkeypatch):
    _fake_runtime(monkeypatch, resolved=True, cuda=False)
    _win, line, _bar, _tp = _window()
    line.refresh_runtime()
    assert line.info.text() == "CPU"
    assert "CUDA 12.8" not in line.info.text()


def test_main_window_uses_one_status_surface(app, monkeypatch):
    monkeypatch.setattr(canvas_gl, "canvas_line", lambda: (CANVAS, False))
    from app.gui.main_window import MainWindow
    from app.gui.shell_chrome import install_status_surface

    mw = MainWindow()
    try:
        install_status_surface(mw)  # idempotent: MainWindow may already have done it
        mw.app_stack.setCurrentIndex(1)
        mw.show()
        app.processEvents()
        assert not mw.status_bar.isVisible()
        mw.status_bar.showMessage("Mask saved for a.png", 3000)
        assert mw.statusline_message.text() == "Mask saved for a.png"
        assert mw.task_progress.parentWidget() is mw.statusline
        mw.statusline.refresh_runtime()
        tip = mw.statusline_info.toolTip().splitlines()  # real, not hard-coded
        assert tip[0] == f"Inference runtime: {runtime_label()}"
        assert tip[-1] == f"Canvas: {CANVAS}"
        assert mw.statusline.height() == 28
    finally:
        delete_widgets([mw])


def _folder(root, name: str, count: int) -> str:
    stems = [f"24EFCAM1_T001_L0{10 + i}_2024.07.18_1115{10 + i}_001_SMM" for i in range(count)]
    return str(write_photos(root / name, stems))


def test_the_idle_text_follows_the_loaded_folder(app, tmp_path, no_prompts):
    window = build_window(tmp_path / "s.ini", size=None, show=False)
    try:
        window.show()
        window.open_path(_folder(tmp_path, "a", 3))
        assert _wait(lambda: len(window.library_index.leaves()) == 3)
        assert _wait(lambda: window.statusline_message.text() == "Ready · 3 images")

        window.open_path(_folder(tmp_path, "b", 5))
        assert _wait(lambda: len(window.library_index.leaves()) == 5)
        assert _wait(lambda: window.statusline_message.text() == "Ready · 5 images")
    finally:
        delete_widgets([window])


def test_a_failing_idle_text_leaves_ready(app):
    def broken() -> str:
        raise RuntimeError("no library yet")

    _win, line, bar, _tp = _window(idle=broken)
    bar.showMessage("x")
    bar.clearMessage()
    assert line.message.text() == "Ready"
