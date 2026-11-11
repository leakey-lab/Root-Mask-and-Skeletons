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
from pathlib import Path

import pytest

from tests.native_dashboard_support import (
    CHILD_TIMEOUT_S,
    MAIN_QWINDOW,
    STEPS,
    child_command,
)
from tests.native_dashboard_support import recreations as _recreations

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "docs" / "examples"
OPT_IN = "SPROUTS_NATIVE_PLATFORM_PROBE"


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


# The child process (tests/native_dashboard_support.run_child, mode "probe") records the
# window through every step; the tests below judge the record.
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
    done = subprocess.run(
        child_command(project, out, "probe"),
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
