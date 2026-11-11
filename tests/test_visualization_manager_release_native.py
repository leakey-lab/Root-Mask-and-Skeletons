"""The dashboards' 5-minute release on the real Windows platform (dashboard flicker, Part A).

Qt cannot change the surface type of a window that is on screen: a dashboard's browser draws
through Direct3D 11, the canvases through OpenGL, so each switch recreates the main window
(it is gone for 43-98 ms). At the release the dashboards used to leave the window only after
OpenGL came back; Qt then found their browsers first, made the window a Direct3D one again
and an OpenGL one at their deferred delete: two recreations. A released dashboard now leaves
the window first (``DashboardController._discard``): one recreation, ending on OpenGL.

The offscreen platform of the suite never recreates a window, so a child process runs the
real window on the native platform: frameless, never activated, placed off the screen (it
never appears and never takes the focus). The canvases are forced onto Qt's software OpenGL
(``opengl32sw.dll``, shipped with PyQt6) in place of a GPU. Skipped off Windows, and when the
child cannot set up (no window, no OpenGL, a dashboard that does not load).

On Windows each dashboard is now a native child window with its own Direct3D 11 surface
(route (b) of the dashboard blink, KNOWN_ISSUES D8): the main window keeps the canvases'
OpenGL surface, so no step recreates it, and the canvases keep OpenGL while a dashboard lives.
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

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="the native Windows platform")


# The child process (tests/native_dashboard_support.run_child, mode "windows") records the
# window through every step; the tests below judge the record.
def _child_env(tmp_path: Path) -> dict[str, str]:
    env = dict(os.environ)
    # The native platform and the app's own graphics choices (main.py sets them).
    for name in ("QT_QPA_PLATFORM", "QT_QUICK_BACKEND", "QT_WIDGETS_RHI", "QSG_RHI_BACKEND"):
        env.pop(name, None)
    env.update(
        QT_LOGGING_RULES="qt.widgets.painting.debug=true",  # Qt's "Recreating ..." line
        QT_OPENGL="software",  # opengl32sw.dll: the same OpenGL on every machine
        ROOT_VIEWER_ENABLE_OPENGL_VIEWPORT="1",
        SPROUTS_CONFIG_DIR=str(tmp_path / "config"),
        SPROUTS_LOG_DIR=str(tmp_path / "logs"),
        PYTHONIOENCODING="utf-8",
    )
    return env


@pytest.fixture(scope="module")
def release_record(tmp_path_factory) -> dict:
    tmp_path = tmp_path_factory.mktemp("release_native")
    project = tmp_path / "project"
    shutil.copytree(EXAMPLES, project)
    out = tmp_path / "record.json"
    try:
        done = subprocess.run(
            child_command(project, out, "windows"),
            cwd=str(tmp_path),
            env=_child_env(tmp_path),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=CHILD_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        pytest.skip(f"the native child did not finish within {CHILD_TIMEOUT_S} s")
    if not out.exists():
        tail = (done.stderr or done.stdout or "")[-2000:]
        pytest.skip(
            f"the native platform could not run the window (exit {done.returncode}): {tail}"
        )
    record = json.loads(out.read_text(encoding="utf-8"))
    assert "error" not in record, record["error"]
    if record.get("setup_error"):
        pytest.skip(f"native set-up: {record['setup_error']}")
    return record


def test_the_release_ends_on_an_opengl_window_with_opengl_canvases(release_record):
    after = release_record["steps"]["release"]["after"]
    assert release_record["steps"]["release"]["dashboards_left"] == []
    assert after["surface"] == "OpenGLSurface"
    assert after["gl_viewports"] == [True, True]  # the Library display and the Correct canvas
    assert after["visible_top_levels"] == 1  # the window only: no dashboard became a window


def test_no_graphics_api_warning_at_the_release(release_record):
    warnings = [m["text"] for m in release_record["messages"] if "not compatible" in m["text"]]
    assert warnings == []


def test_the_dashboards_open_again_after_the_release(release_record):
    assert release_record["steps"]["reopen"]["ok"]
    assert release_record["steps"]["reopen"]["after"]["gl_viewports"] == [True, True]


#: The steps after the window and the project are shown, in their order.
def test_no_step_recreates_the_main_window(release_record):
    recreations = {step: _recreations(release_record, step) for step in STEPS}
    assert recreations == dict.fromkeys(STEPS, 0)
    recreating = [m for m in release_record["messages"] if "Recreating QWidgetWindow" in m["text"]]
    assert recreating == []


def test_the_main_window_keeps_its_native_window_and_stays_the_only_one(release_record):
    first = release_record["steps"]["show"]["after"]
    for step in STEPS:
        after = release_record["steps"][step]["after"]
        assert (after["qwindow"], after["winid"]) == (first["qwindow"], first["winid"]), step
        assert after["visible_top_levels"] == 1, step
    shown = [e["window"] for e in release_record["events"] if e["event"] == "Show"]
    assert shown == [MAIN_QWINDOW]  # no dashboard ever became a window of its own


def test_the_canvases_keep_opengl_while_a_dashboard_lives(release_record):
    for step in ("length", "area", "visualize_again", "reopen"):
        after = release_record["steps"][step]["after"]
        assert after["surface"] == "OpenGLSurface", step
        assert after["gl_viewports"] == [True, True], step
