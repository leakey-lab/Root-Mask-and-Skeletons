"""The start-up device check (``app/gui/device_check.py``).

:class:`DeviceCheck` names the inference device once per window, without loading torch: its
pool job (:func:`resolve_device_logged`) asks the NVIDIA driver and reads the torch build
while the runtime is unresolved, and asks nothing once torch has resolved it. It starts only
after the window shows, once the user's input (mouse, keys, wheel, drag and drop, anywhere in
the app) has paused for ``idle_ms`` and no task runs; ``SPROUTS_DEVICE_CHECK=0`` or the
setting ``device_check/at_startup = false`` turns it off (the environment wins; the chip is
then left alone). Whatever ends the job, a ``BaseException`` included, the chip is told
(never stuck at "Checking GPU…"); a cancelled check leaves no job and no filter. Headless
windows skip the check unless asked.

The resolver is a stub or the runtime module a fake: torch is never loaded here.
"""

from __future__ import annotations

import sys
import threading
import time
import types

import pytest
from PyQt6.QtCore import QEvent, QPointF, Qt, QThreadPool
from PyQt6.QtGui import QKeyEvent, QMouseEvent, QWheelEvent
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QWidget

from app.gui import device_check, shell_chrome
from app.gui.device_check import DeviceCheck, device_check_enabled
from app.gui.shell_chrome import DEVICE_CHECKING
from app.inference import device_probe
from app.inference.device_probe import DeviceGuess
from tests.device_chip_support import GPU, RUNTIME, FakeRuntime, GuessLine, Line
from tests.qt_teardown import delete_widgets, flush_deleted
from tests.qt_wait import settle
from tests.qt_wait import wait_until as wait
from tests.window_support import build_window, quiet_window

pytestmark = pytest.mark.usefixtures("no_prompts", "slot_errors")


class Settings:
    def __init__(self, **values) -> None:
        self.values = values

    def value(self, key, default=None):
        return self.values.get(key, default)


class Resolver:
    """The pool job: records when (perf_counter) it ran."""

    def __init__(self) -> None:
        self.times: list[float] = []

    def __call__(self) -> str:
        self.times.append(time.perf_counter())
        return ""


@pytest.fixture
def widgets(app):
    """Widgets to delete after the test (skips the ones a test deleted); the pool's jobs end
    first."""
    made: list = []
    yield made
    delete_widgets(made)
    assert QThreadPool.globalInstance().waitForDone(5000)
    flush_deleted()


@pytest.fixture
def window(widgets):
    def make():
        widget = QWidget()
        widget.resize(200, 120)
        widgets.append(widget)
        return widget

    return make


@pytest.fixture
def widget(window):
    return window()


@pytest.fixture
def runtime(monkeypatch):
    fake = FakeRuntime()
    monkeypatch.setitem(sys.modules, RUNTIME, fake.module)
    import app.inference as inference_package

    monkeypatch.setattr(inference_package, "runtime", fake.module, raising=False)
    return fake


@pytest.fixture
def make_window(app, tmp_path):
    """``make(**kwargs)``: a MainWindow on its own INI settings, neither resized nor shown."""
    made = []

    def make(**kwargs):
        path = tmp_path / f"settings_{len(made)}.ini"
        made.append(build_window(path, size=None, show=False, **kwargs))
        return made[-1]

    yield make
    assert QThreadPool.globalInstance().waitForDone(5000)
    delete_widgets(made, prepare=quiet_window)
    QApplication.processEvents()


def pump(ms: int) -> None:
    QTest.qWait(ms)


def move(target: QWidget, i: int = 0) -> None:
    point = QPointF(10 + i % 50, 10)
    QApplication.sendEvent(
        target,
        QMouseEvent(
            QEvent.Type.MouseMove,
            point,
            point,
            Qt.MouseButton.NoButton,
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
        ),
    )


class Abort(BaseException):
    """Neither an ``Exception`` nor ``SystemExit`` / ``KeyboardInterrupt``."""


ENDINGS = [SystemExit(3), KeyboardInterrupt(), Abort("stopped")]
ENDING_IDS = ["exit", "interrupt", "other"]


def raiser(ending: BaseException):
    def raise_it(*_args, **_kwargs):
        raise ending

    return raise_it


def start_check(widgets, resolve=None) -> Line:
    widget, line = QWidget(), Line()
    widgets.append(widget)
    DeviceCheck(widget, line, delay_ms=10, idle_ms=20, resolve=resolve, enabled=True)
    assert line.calls == [(shell_chrome.CHECKING, "")]
    widget.show()
    return line


# ------------------------------------------------------------------ the opt-out
@pytest.mark.parametrize("value", ["0", "false", "No", " off "])
def test_the_environment_turns_the_check_off(value):
    assert device_check_enabled(Settings(), {"SPROUTS_DEVICE_CHECK": value}) is False


@pytest.mark.parametrize("env", [{}, {"SPROUTS_DEVICE_CHECK": "1"}, {"SPROUTS_DEVICE_CHECK": ""}])
def test_the_check_is_on_by_default(env):
    assert device_check_enabled(Settings(), env) is True


@pytest.mark.parametrize("stored", [False, "false", "0", 0])
def test_the_setting_turns_the_check_off(stored):
    settings = Settings(**{"device_check/at_startup": stored})
    assert device_check_enabled(settings, {}) is False


def test_the_environment_wins_over_the_setting():
    off = Settings(**{"device_check/at_startup": "false"})
    on = Settings(**{"device_check/at_startup": True})
    assert device_check_enabled(off, {"SPROUTS_DEVICE_CHECK": "1"}) is True
    assert device_check_enabled(on, {"SPROUTS_DEVICE_CHECK": "0"}) is False


def test_an_unreadable_setting_keeps_the_check_on():
    class Broken:
        def value(self, key, default=None):
            raise OSError("registry unavailable")

    assert device_check_enabled(Settings(**{"device_check/at_startup": "maybe"}), {}) is True
    assert device_check_enabled(Broken(), {}) is True


def test_a_check_that_is_off_never_runs_nor_touches_the_chip(app, window):
    widget, line, resolve = window(), Line(), Resolver()
    check = DeviceCheck(widget, line, delay_ms=10, idle_ms=30, resolve=resolve, enabled=False)
    widget.show()
    pump(250)
    assert resolve.times == []
    assert line.calls == []  # the chip is left alone (no "Checking GPU…")
    assert not check.started() and not check.watching_input()


def test_the_window_passes_its_settings_opt_out(app, window, monkeypatch):
    monkeypatch.delenv("SPROUTS_DEVICE_CHECK", raising=False)
    widget, line, resolve = window(), Line(), Resolver()
    settings = Settings(**{"device_check/at_startup": "false"})
    DeviceCheck(widget, line, delay_ms=10, idle_ms=30, resolve=resolve, settings=settings)
    widget.show()
    pump(250)
    assert resolve.times == [] and line.calls == []


def test_headless_windows_skip_the_check_by_default(app, make_window, runtime, monkeypatch):
    calls = []
    monkeypatch.setattr(device_check, "resolve_device_logged", lambda: calls.append(1))
    window = make_window()
    window.show()
    QTest.qWait(device_check.DEVICE_CHECK_DELAY_MS + 200)
    settle()
    assert calls == []
    assert window.statusline_info.text() == "Device not detected yet"


# ---------------------------------------------------------------- the schedule
def test_without_input_the_check_starts_once_the_idle_period_passed(app, window):
    widget, line, resolve = window(), Line(), Resolver()
    check = DeviceCheck(widget, line, delay_ms=10, idle_ms=200, resolve=resolve, enabled=True)
    assert line.calls == [("checking", "")]
    pump(300)
    assert resolve.times == []  # never before the window shows
    shown = time.perf_counter()
    widget.show()
    assert check.watching_input()
    assert wait(lambda: resolve.times)
    assert resolve.times[0] - shown >= 0.18
    assert wait(lambda: line.calls[-1] == (None, ""))
    assert not check.watching_input()  # the application filter is gone


def test_the_check_waits_while_the_user_keeps_moving_the_mouse(app, window):
    widget, line, resolve = window(), Line(), Resolver()
    DeviceCheck(widget, line, delay_ms=10, idle_ms=250, resolve=resolve, enabled=True)
    widget.show()
    other = QWidget()  # input anywhere in the app counts, not only on the watched window
    try:
        for i in range(16):  # 0.8 s of mouse movement, every 50 ms
            move(other if i % 2 else widget, i)
            last_input = time.perf_counter()
            pump(50)
            assert resolve.times == [], f"started during the input (step {i})"
        assert wait(lambda: resolve.times)
        assert resolve.times[0] - last_input >= 0.24  # the whole quiet period (250 ms)
    finally:
        other.deleteLater()
        flush_deleted()


@pytest.mark.parametrize("kind", ["key", "click", "wheel"])
def test_keys_clicks_and_wheel_count_as_input(app, window, kind):
    widget, line, resolve = window(), Line(), Resolver()
    # The quiet period is well beyond the 100 ms between the events, under load too.
    DeviceCheck(widget, line, delay_ms=10, idle_ms=600, resolve=resolve, enabled=True)
    widget.show()
    point = QPointF(5, 5)
    for _ in range(8):
        if kind == "key":
            event = QKeyEvent(
                QEvent.Type.KeyPress, Qt.Key.Key_PageDown, Qt.KeyboardModifier.NoModifier
            )
        elif kind == "click":
            event = QMouseEvent(
                QEvent.Type.MouseButtonPress,
                point,
                point,
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
            )
        else:
            event = QWheelEvent(
                point,
                point,
                QPointF(0, 0).toPoint(),
                QPointF(0, 120).toPoint(),
                Qt.MouseButton.NoButton,
                Qt.KeyboardModifier.NoModifier,
                Qt.ScrollPhase.NoScrollPhase,
                False,
            )
        QApplication.sendEvent(widget, event)
        pump(100)
        assert resolve.times == []
    assert wait(lambda: resolve.times)


def test_a_running_task_defers_the_check(app, window):
    widget, line, resolve = window(), Line(), Resolver()
    line.busy = True  # e.g. the first folder is loading
    DeviceCheck(widget, line, delay_ms=10, idle_ms=100, resolve=resolve, enabled=True)
    widget.show()
    pump(450)
    assert resolve.times == []
    line.busy = False
    assert wait(lambda: resolve.times)


def test_input_after_the_start_is_not_watched_and_starts_nothing_twice(app, window):
    widget, line, resolve = window(), Line(), Resolver()
    check = DeviceCheck(widget, line, delay_ms=10, idle_ms=50, resolve=resolve, enabled=True)
    widget.show()
    assert wait(lambda: resolve.times)
    assert not check.watching_input()
    for i in range(5):
        move(widget, i)
    widget.hide()
    widget.show()  # a second show does not check again
    pump(200)
    assert len(resolve.times) == 1 and check.started()


def test_a_window_closed_before_the_check_leaves_no_job_and_no_filter(app, window):
    widget, line, resolve = window(), Line(), Resolver()
    DeviceCheck(widget, line, delay_ms=10, idle_ms=150, resolve=resolve, enabled=True)
    widget.show()
    pump(30)
    widget.close()
    widget.deleteLater()
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    other = QWidget()
    try:
        move(other)  # the application filter of the deleted check must not be called
        pump(300)
    finally:
        other.deleteLater()
    assert resolve.times == []


def test_the_check_runs_its_job_off_the_gui_thread(app, window):
    """The job asks the driver (about 20 ms, more on a cold disk): never on the GUI thread."""
    widget, line, threads = window(), Line(), []
    check = DeviceCheck(
        widget,
        line,
        delay_ms=10,
        idle_ms=20,
        resolve=lambda: threads.append(threading.get_ident()) or "",
        enabled=True,
    )
    widget.show()
    assert wait(lambda: line.calls[-1] == (None, ""))
    assert len(threads) == 1 and threads[0] != threading.main_thread().ident
    assert check.started()


def test_a_cancelled_check_never_starts(app, widget):
    line = GuessLine()
    runs = []
    check = DeviceCheck(
        widget, line, delay_ms=10, idle_ms=20, resolve=lambda: runs.append(1), enabled=True
    )
    widget.show()
    check.cancel()  # the window closes before the check started
    QTest.qWait(150)
    assert runs == [] and not check.watching_input()


# ------------------------------------------------------------------ the job
def test_the_job_asks_the_driver_while_the_runtime_is_unresolved(runtime, monkeypatch):
    monkeypatch.setattr(device_probe, "probe_device", lambda: GPU)
    assert device_check.resolve_device_logged() is GPU


def test_the_job_asks_nothing_once_the_runtime_is_resolved(runtime, monkeypatch):
    runtime.describe(resolve=True)

    def probe():
        raise AssertionError("the driver was asked although torch knows the device")

    monkeypatch.setattr(device_probe, "probe_device", probe)
    assert device_check.resolve_device_logged() == ""


def test_resolve_device_logged_reports_an_import_failure(monkeypatch, caplog):
    broken = types.ModuleType(RUNTIME)

    def describe_runtime(resolve: bool = False):
        raise ImportError("DLL load failed while importing _C")

    broken.describe_runtime = describe_runtime
    monkeypatch.setitem(sys.modules, RUNTIME, broken)
    import app.inference as inference_package

    monkeypatch.setattr(inference_package, "runtime", broken, raising=False)
    assert device_check.resolve_device_logged() == "DLL load failed while importing _C"
    assert "Could not detect the inference device" in caplog.text


# ------------------------------------------------------------------ what the chip is told
def test_the_check_hands_its_guess_to_the_chip(app, widget):
    line = GuessLine()
    DeviceCheck(widget, line, delay_ms=10, idle_ms=20, resolve=lambda: GPU, enabled=True)
    widget.show()
    assert wait(lambda: line.calls[-1] == ("guess", GPU))
    assert line.calls[0] == ("check", "checking", "")


def test_no_torch_is_a_failed_check(app, widget):
    line = GuessLine()
    missing = DeviceGuess("unknown", problem="PyTorch is not installed: Generate cannot run.")
    DeviceCheck(widget, line, delay_ms=10, idle_ms=20, resolve=lambda: missing, enabled=True)
    widget.show()
    assert wait(lambda: line.calls[-1][:2] == ("check", "failed"))
    assert "not installed" in line.calls[-1][2]


@pytest.mark.parametrize("ending", ENDINGS, ids=ENDING_IDS)
def test_a_check_ended_by_a_base_exception_still_resolves_the_chip(widgets, ending, caplog):
    line = start_check(widgets, resolve=raiser(ending))

    assert wait(lambda: len(line.calls) == 2), "the chip stayed at 'Checking GPU…'"
    state, detail = line.calls[-1]
    assert state == shell_chrome.FAILED and detail  # "Device unknown", and why
    assert "Could not detect the inference device" in caplog.text


def test_the_window_chip_shows_the_driver_guess(app, runtime, monkeypatch, tmp_path):
    monkeypatch.setattr(device_probe, "probe_device", lambda: GPU)
    made = [build_window(tmp_path / "settings.ini", size=None, show=False, detect_device=True)]
    window = made[0]
    try:
        assert window.statusline_info.text() == DEVICE_CHECKING
        window.show()
        tip = window.statusline_info.toolTip
        assert wait(lambda: tip().startswith("Inference runtime: GPU · Fake RTX 9000 · CUDA 12.8"))
        assert "Named by the NVIDIA driver." in tip()
    finally:
        assert QThreadPool.globalInstance().waitForDone(5000)
        delete_widgets(made)
