"""The child process of the native-platform dashboard probes (KNOWN_ISSUES D8).

``run_child`` drives the real main window on the native platform through two dashboards, the
5-minute release and a reopen, and writes a record (JSON) of every native surface event of a
top-level window, every "Recreating QWidgetWindow" / "not compatible" message and the window's
state after each step. It is the driver of
``tests/test_visualization_manager_release_native.py`` (``mode="windows"``: the canvases forced
onto software OpenGL, the window off the screen) and ``tests/test_dashboard_native_platform.py``
(``mode="probe"``: Linux and macOS, plus the renderer's sandbox facts). Holds no assert: the
test modules judge the record.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
#: Qt names the main window's native window after its class.
MAIN_QWINDOW = "MainWindowClassWindow"
CHILD_TIMEOUT_S = 300
#: Longest wait for a dashboard page, a project load or the release to settle (child).
LOAD_TIMEOUT_MS = 120_000
SETTLE_MS = 2500
#: The steps after the window and the project are shown, in their order.
STEPS = ("length", "area", "library", "visualize_again", "library_again", "release", "reopen")


def child_command(project, out, mode: str) -> list[str]:
    """The command that runs :func:`run_child` in a fresh interpreter."""
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "from tests.native_dashboard_support import run_child; "
        "run_child(sys.argv[2], sys.argv[3], sys.argv[4])"
    )
    return [sys.executable, "-c", code, str(ROOT), str(project), str(out), mode]


def recreations(record: dict, step: str) -> int:
    """How often the main window's native window was destroyed during ``step``."""
    return sum(
        1
        for event in record["events"]
        if event["step"] == step
        and event["window"] == MAIN_QWINDOW
        and event["event"] == "SurfaceAboutToBeDestroyed"
    )


def run_child(project: str, out: str, mode: str) -> None:
    """Drive the window through two dashboards, the release and a reopen; write the record
    to ``out`` (JSON). Runs in its own process on the native platform. ``mode``: "windows"
    or "probe" (see the module docstring)."""
    probe = mode == "probe"
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
    record: dict = {"steps": {}, "events": events, "messages": messages}
    if probe:
        record["dashboards"] = {}

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
    name = "sprouts-native-platform" if probe else "sprouts-release-native"
    app = QApplication.instance() or QApplication([name])
    app_main.apply_stylesheet(app)
    recorder = Recorder()
    app.installEventFilter(recorder)
    if probe:
        record["platform"] = QGuiApplication.platformName()
        record["sandbox_disabled"] = os.environ.get("QTWEBENGINE_DISABLE_SANDBOX", "")
        record["euid"] = os.geteuid() if hasattr(os, "geteuid") else None
        try:
            with open(
                "/proc/sys/kernel/apparmor_restrict_unprivileged_userns", encoding="utf-8"
            ) as f:
                record["apparmor_restrict_unprivileged_userns"] = f.read().strip()
        except OSError:
            record["apparmor_restrict_unprivileged_userns"] = None

    from app.gui import canvas_gl
    from app.gui import visualization_manager as vm
    from app.gui.main_window import MainWindow
    from app.visualization.viz_window import native_child_surface

    found = canvas_gl.opengl_support()
    if probe:
        record["opengl"] = {"supported": found.supported, "reason": found.reason}
        surface = native_child_surface()
        record["native_child_surface"] = surface.name if surface is not None else None
    else:
        if not found.checked or found.reason.startswith("no OpenGL") or "below 2.0" in found.reason:
            record["setup_error"] = f"no OpenGL context: {found.reason}"
            write()
            return
        record["opengl"] = found.renderer or found.reason
        # A GPU's path: the canvases use QOpenGLWidget viewports (software OpenGL counts as
        # unsupported otherwise).
        canvas_gl.opengl_support = lambda: canvas_gl.GLSupport(True, "forced by the test", "")

    settings = QSettings(
        os.path.join(os.path.dirname(out), "settings.ini"), QSettings.Format.IniFormat
    )
    window = MainWindow(settings=settings, detect_device=False, warm_up=False)
    if probe:
        # Never take the keyboard from a desktop's user. Qt::Tool hides with an inactive app
        # on macOS, so the window stays a normal one there.
        if sys.platform.startswith("linux"):
            window.setWindowFlags(Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint)
        window.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        window.setGeometry(40, 40, 1240, 780)
    else:
        window.setWindowFlags(Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint)
        window.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        window.setGeometry(-6000, -6000, 1240, 780)
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
        if probe and board is not None:
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
        if not probe and not all(record["steps"]["show"]["after"]["gl_viewports"]):
            record["setup_error"] = "the canvases did not get OpenGL viewports"
            return
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
