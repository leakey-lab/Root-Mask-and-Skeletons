"""The dashboard blink (KNOWN_ISSUES D8) on the native Linux and macOS platforms.

Qt cannot change the surface type of a window on screen. A dashboard's browser draws through a
``QQuickWidget``, which needs an RHI surface (OpenGL on Linux, Metal on macOS): a main window made
for raster canvases is destroyed and made again when the first dashboard joins it as a plain
child widget (a blink). ``test_visualization_manager_release_native.py`` pins Windows; this file
asks the same question on Linux (X11, Wayland) and macOS (Cocoa). A child process runs the real
window on the native platform through two dashboards, the 5-minute release and a reopen, and
records every native surface the main window gets, plus (Linux) the renderer's sandbox.

Opt-in: it runs only with ``SPROUTS_NATIVE_PLATFORM_PROBE`` set (``1`` for the platform's own
default, or a Qt platform name such as ``xcb`` or ``wayland``), never on Windows and never on
Linux without a display. CI runs it in its own step: on macOS in the runner's Cocoa session, on
Linux under xvfb (``.github/workflows/ci.yml``). The record is printed, so ``pytest -s`` shows the
answer in the log even when an assertion fails.

Measured on 2026-10-29 in WSL Ubuntu 24.04: X11 and Wayland recreated the main window once per
app start, at the first dashboard, with raster canvases, and never with OpenGL canvases. On X11
each dashboard is now a native OpenGL child (``viz_window.native_child_surface``), so no step
recreates it there; on Wayland (plain children) this probe still fails with raster canvases.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "docs" / "examples"
OPT_IN = "SPROUTS_NATIVE_PLATFORM_PROBE"
#: Qt names the main window's native window after its class.
MAIN_QWINDOW = "MainWindowClassWindow"
CHILD_TIMEOUT_S = 300
LOAD_TIMEOUT_MS = 120_000
SETTLE_MS = 2500
#: The steps after the window and the project are shown, in their order.
STEPS = ("length", "area", "library", "visualize_again", "library_again", "release", "reopen")


def _skip_reason() -> str | None:
    if sys.platform == "win32":
        return "Windows: test_visualization_manager_release_native.py"
    if not os.environ.get(OPT_IN):
        return f"the native-platform probe runs with {OPT_IN}=1 (CI's own step)"
    if sys.platform.startswith("linux") and not (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    ):
        return "no display (run it under xvfb-run)"
    return None


pytestmark = pytest.mark.skipif(_skip_reason() is not None, reason=_skip_reason() or "")


# --------------------------------------------------------------------------- the child
def run_child(project: str, out: str) -> None:
    """Drive the window through two dashboards, the release and a reopen; write the record
    to ``out`` (JSON). Runs in its own process on the native platform."""
    sys.path.insert(0, str(ROOT))
    # First: the app's Qt / Chromium environment, logging and QtWebEngine, before Qt loads.
    import main as app_main

    # isort: split
    from PyQt6 import sip
    from PyQt6.QtCore import (
        QEvent,
        QEventLoop,
        QObject,
        QSettings,
        Qt,
        QTimer,
        qInstallMessageHandler,
    )
    from PyQt6.QtGui import QGuiApplication, QWindow
    from PyQt6.QtOpenGLWidgets import QOpenGLWidget
    from PyQt6.QtWidgets import QApplication

    step = ["start"]
    events: list[dict] = []
    messages: list[dict] = []
    record: dict = {"steps": {}, "events": events, "messages": messages, "dashboards": {}}

    def on_message(_mode, _context, text: str) -> None:
        if "Recreating QWidgetWindow" in text or "not compatible" in text:
            messages.append({"step": step[0], "text": text[:300]})

    class Recorder(QObject):
        def eventFilter(self, obj, event):  # noqa: N802 - Qt's name
            kind = event.type()
            if kind == QEvent.Type.PlatformSurface and isinstance(obj, QWindow):
                if obj.parent() is None:
                    events.append(
                        {
                            "step": step[0],
                            "event": event.surfaceEventType().name,
                            "window": obj.objectName(),
                            "surface": obj.surfaceType().name,
                        }
                    )
            elif kind == QEvent.Type.Show and isinstance(obj, QWindow) and obj.parent() is None:
                events.append({"step": step[0], "event": "Show", "window": obj.objectName()})
            return False

    def wait(ms: int) -> None:
        loop = QEventLoop()
        QTimer.singleShot(ms, loop.quit)
        loop.exec()

    def wait_until(predicate, ms: int) -> bool:
        end = time.monotonic() + ms / 1000
        while time.monotonic() < end:
            if predicate():
                return True
            wait(25)
        return predicate()

    def write() -> None:
        with open(out, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=1)

    qInstallMessageHandler(on_message)
    app = QApplication.instance() or QApplication(["sprouts-native-platform"])
    app_main.apply_stylesheet(app)
    recorder = Recorder()
    app.installEventFilter(recorder)
    record["platform"] = QGuiApplication.platformName()
    record["sandbox_disabled"] = os.environ.get("QTWEBENGINE_DISABLE_SANDBOX", "")
    record["euid"] = os.geteuid() if hasattr(os, "geteuid") else None
    try:
        with open("/proc/sys/kernel/apparmor_restrict_unprivileged_userns", encoding="utf-8") as f:
            record["apparmor_restrict_unprivileged_userns"] = f.read().strip()
    except OSError:
        record["apparmor_restrict_unprivileged_userns"] = None

    from app.gui import canvas_gl
    from app.gui import visualization_manager as vm
    from app.gui.main_window import MainWindow
    from app.visualization.viz_window import native_child_surface

    found = canvas_gl.opengl_support()
    record["opengl"] = {"supported": found.supported, "reason": found.reason}
    surface = native_child_surface()
    record["native_child_surface"] = surface.name if surface is not None else None

    settings = QSettings(
        os.path.join(os.path.dirname(out), "settings.ini"), QSettings.Format.IniFormat
    )
    window = MainWindow(settings=settings, detect_device=False, warm_up=False)
    # Never take the keyboard from a desktop's user. Qt::Tool hides with an inactive app on
    # macOS, so the window stays a normal one there.
    if sys.platform.startswith("linux"):
        window.setWindowFlags(Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint)
    window.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
    window.setGeometry(40, 40, 1240, 780)
    views = (window.display_controller.magnifying_view, window.correct_editor.graphics_view)

    def state() -> dict:
        handle = window.windowHandle()
        return {
            "qwindow": sip.unwrapinstance(handle) if handle is not None else 0,
            "winid": int(window.winId()),
            "surface": handle.surfaceType().name if handle is not None else None,
            "gl_viewports": [isinstance(view.viewport(), QOpenGLWidget) for view in views],
            "visible_top_levels": sum(
                1 for w in QGuiApplication.topLevelWindows() if w.isVisible()
            ),
        }

    def sandbox(board) -> dict:
        """The renderer's sandbox on Linux: seccomp mode and whether it has namespaces of its
        own (Chromium's two layers)."""
        pid = board.web_view.page().renderProcessPid()
        facts: dict = {"renderer_pid": pid}
        if not sys.platform.startswith("linux") or not pid:
            return facts
        try:
            for ns in ("user", "pid", "net"):
                own = os.readlink(f"/proc/self/ns/{ns}")
                facts[f"own_{ns}_namespace"] = own != os.readlink(f"/proc/{pid}/ns/{ns}")
            with open(f"/proc/{pid}/status", encoding="utf-8") as f:
                for line in f:
                    key, _, value = line.partition(":")
                    if key in ("Seccomp", "NoNewPrivs"):
                        facts[key] = value.strip()
        except OSError as exc:
            facts["error"] = str(exc)
        return facts

    def dashboard_shown(spec=None) -> bool:
        """``spec``'s dashboard (default: the one on screen) has loaded its page."""
        controller = vm.controller_for(window)

        def current():
            kind = spec.kind if spec is not None else controller.visible_kind()
            return controller.dashboard(vm.SPECS[kind]) if kind else None

        loaded = wait_until(
            lambda: (
                (board := current()) is not None
                and (board.failed or board.loading_label.isHidden())
            ),
            LOAD_TIMEOUT_MS,
        )
        board = current()
        if board is not None:
            record["dashboards"][step[0]] = {
                "failed": board.failed,
                "status": board.loading_label.text()[:500] if board.failed else "",
                "sandbox": sandbox(board) if loaded and not board.failed else {},
            }
        return loaded and board is not None and not board.failed

    def run_step(name: str, action, check=None) -> bool:
        step[0] = name
        record["steps"][name] = {"before": state()}
        action()
        ok = check() if check is not None else True
        wait(SETTLE_MS)
        record["steps"][name]["after"] = state()
        record["steps"][name]["ok"] = ok
        return ok

    try:
        step[0] = "show"
        window.show()
        wait(800)
        window.open_path(project)
        if not wait_until(
            lambda: not window._loading_project and window.loading_overlay.isHidden(),
            LOAD_TIMEOUT_MS,
        ):
            record["setup_error"] = "the project did not load"
            return
        wait(SETTLE_MS)
        record["steps"]["show"] = {"after": state()}
        stage = window.ribbon_buttons
        controller = vm.controller_for(window)
        if not run_step(
            "length", lambda: stage["Visualize"].click(), lambda: dashboard_shown(vm.LENGTH)
        ):
            record["setup_error"] = "the length dashboard did not load"
            return
        if not run_step(
            "area",
            lambda: window.ribbon.viz_seg.button("area").click(),
            lambda: dashboard_shown(vm.AREA),
        ):
            record["setup_error"] = "the area dashboard did not load"
            return
        run_step("library", lambda: stage["Library"].click())
        run_step("visualize_again", lambda: stage["Visualize"].click())
        run_step("library_again", lambda: stage["Library"].click())

        def release() -> None:  # what the 5-minute countdown runs when it ends
            controller._idle_timer.stop()
            controller._on_idle()

        run_step("release", release)
        record["steps"]["release"]["dashboards_left"] = [
            spec.kind for spec in vm.SPECS.values() if controller.dashboard(spec) is not None
        ]
        run_step("reopen", lambda: stage["Visualize"].click(), dashboard_shown)
    except BaseException:  # noqa: BLE001 - reported to the test, which fails on it
        record["error"] = traceback.format_exc()
    finally:
        step[0] = "end"
        write()
        app.removeEventFilter(recorder)
        qInstallMessageHandler(None)
        try:
            from tests.qt_teardown import delete_widgets

            delete_widgets([window])
        finally:
            sys.stdout.flush()
            os._exit(0)  # QtWebEngine's own teardown is not under test


# --------------------------------------------------------------------------- the test
def _child_env(tmp_path: Path) -> dict[str, str]:
    env = dict(os.environ)
    # The native platform (the suite's conftest sets offscreen) and the app's own graphics
    # choices (main.py sets them).
    for name in ("QT_QPA_PLATFORM", "QT_QUICK_BACKEND", "QT_WIDGETS_RHI", "QSG_RHI_BACKEND"):
        env.pop(name, None)
    chosen = os.environ.get(OPT_IN, "")
    if chosen not in ("", "1"):
        env["QT_QPA_PLATFORM"] = chosen  # e.g. xcb or wayland
    env.update(
        QT_LOGGING_RULES="qt.widgets.painting.debug=true",  # Qt's "Recreating ..." line
        SPROUTS_CONFIG_DIR=str(tmp_path / "config"),
        SPROUTS_LOG_DIR=str(tmp_path / "logs"),
        PYTHONIOENCODING="utf-8",
    )
    return env


def _recreations(record: dict, step: str) -> int:
    """How often the main window's native window was destroyed during ``step``."""
    return sum(
        1
        for event in record["events"]
        if event["step"] == step
        and event["window"] == MAIN_QWINDOW
        and event["event"] == "SurfaceAboutToBeDestroyed"
    )


def _summary(record: dict) -> dict:
    return {
        "platform": record.get("platform"),
        "opengl": record.get("opengl"),
        "native_child_surface": record.get("native_child_surface"),
        "main_window_surface": {
            step: record["steps"].get(step, {}).get("after", {}).get("surface")
            for step in ("show", *STEPS)
        },
        "recreations": {step: _recreations(record, step) for step in STEPS},
        "dashboards": record.get("dashboards"),
        "euid": record.get("euid"),
        "apparmor_restrict_unprivileged_userns": record.get(
            "apparmor_restrict_unprivileged_userns"
        ),
        "setup_error": record.get("setup_error"),
    }


@pytest.fixture(scope="module")
def native_record(tmp_path_factory) -> dict:
    tmp_path = tmp_path_factory.mktemp("native_platform")
    project = tmp_path / "project"
    shutil.copytree(EXAMPLES, project)
    out = tmp_path / "record.json"
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "from tests.test_dashboard_native_platform import run_child; "
        "run_child(sys.argv[2], sys.argv[3])"
    )
    done = subprocess.run(
        [sys.executable, "-c", code, str(ROOT), str(project), str(out)],
        cwd=str(tmp_path),
        env=_child_env(tmp_path),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=CHILD_TIMEOUT_S,
    )
    if not out.exists():
        tail = (done.stderr or done.stdout or "")[-3000:]
        pytest.fail(
            f"the native platform could not run the window (exit {done.returncode}):\n{tail}"
        )
    record = json.loads(out.read_text(encoding="utf-8"))
    print("\nnative-platform dashboard probe:", json.dumps(_summary(record), indent=1))
    assert "error" not in record, record["error"]
    return record


def test_the_dashboards_load_on_the_native_platform(native_record):
    assert native_record.get("setup_error") is None, native_record.get("dashboards")
    assert native_record["steps"]["reopen"]["ok"]


def test_no_step_recreates_the_main_window(native_record):
    recreations = {step: _recreations(native_record, step) for step in STEPS}
    assert recreations == dict.fromkeys(STEPS, 0)
    recreating = [m for m in native_record["messages"] if "Recreating QWidgetWindow" in m["text"]]
    assert recreating == []


def test_the_main_window_keeps_its_native_window_and_stays_the_only_one(native_record):
    first = native_record["steps"]["show"]["after"]
    for step in STEPS:
        after = native_record["steps"][step]["after"]
        assert (after["qwindow"], after["winid"]) == (first["qwindow"], first["winid"]), step
        assert after["visible_top_levels"] == 1, step
    shown = [e["window"] for e in native_record["events"] if e["event"] == "Show"]
    assert shown == [MAIN_QWINDOW]  # no dashboard ever became a window of its own


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux /proc")
def test_the_renderer_runs_in_chromiums_seccomp_sandbox_as_a_normal_user(native_record):
    """A normal user keeps the Chromium sandbox (it is off only for root, app/qt_environment.py).
    Where user namespaces are unusable (Ubuntu 24.04's AppArmor restriction) the renderer has no
    namespaces of its own but keeps the seccomp-bpf filter; the summary above says which."""
    if native_record["euid"] == 0 or native_record["sandbox_disabled"]:
        pytest.skip("the sandbox is off (root, or QTWEBENGINE_DISABLE_SANDBOX)")
    facts = native_record["dashboards"]["length"]["sandbox"]
    assert facts.get("Seccomp") == "2", facts  # 2: a seccomp-bpf filter is on
