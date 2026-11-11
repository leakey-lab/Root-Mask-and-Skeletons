"""The ``main.py`` entry point: what importing it sets, how ``main()`` starts and ends.

Importing ``main`` sets the Qt start-up flags of this OS and lets torch's idle OpenMP
workers sleep (``KMP_BLOCKTIME=0``, unless the user set one). ``main()`` starts from any
working directory with the theme, icon, names and exception hooks in place; a window that
cannot be built is a critical box and exit code 1, an app module that cannot be imported is
written to the log; once the event loop returns the window is deleted while the
QApplication still exists. The palette's disabled text uses the disabled tier.

``main()`` touches process-global state, so it runs in fresh interpreters with
``SPROUTS_LOG_DIR`` in ``tmp_path`` and a stub window (no model, no data).
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from PyQt6.QtGui import QPalette

REPO_ROOT = Path(os.path.abspath(__file__)).parent.parent


def _run(code: str, *, cwd: Path, env_extra: dict[str, str] | None = None, timeout: int = 180):
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", PYTHONDONTWRITEBYTECODE="1")
    env.pop("SPROUTS_LOG_DIR", None)
    env.update(env_extra or {})
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


# --- importing main ----------------------------------------------------------------
def _import_main(tmp_path, **env_vars: str) -> subprocess.Popen:
    """A fresh process that imports ``main`` and prints ``QSG_RHI_BACKEND KMP_BLOCKTIME``;
    neither is set before the import unless ``env_vars`` sets it."""
    env = dict(
        os.environ,
        QT_QPA_PLATFORM="offscreen",
        PYTHONDONTWRITEBYTECODE="1",
        SPROUTS_LOG_DIR=str(tmp_path),
    )
    for name in ("QSG_RHI_BACKEND", "QTWEBENGINE_CHROMIUM_FLAGS", "KMP_BLOCKTIME"):
        env.pop(name, None)
    env.update(env_vars)
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); import main, os; "
        "print(os.environ.get('QSG_RHI_BACKEND'), os.environ.get('KMP_BLOCKTIME'))"
    )
    return subprocess.Popen(
        [sys.executable, "-c", code, str(REPO_ROOT)],
        cwd=str(tmp_path),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _printed(proc: subprocess.Popen) -> list[str]:
    try:
        out, err = proc.communicate(timeout=180)
    except subprocess.TimeoutExpired:
        proc.kill()
        raise
    assert proc.returncode == 0, err
    return out.strip().splitlines()[-1].split()


@pytest.fixture(scope="module")
def imports_of_main(tmp_path_factory):
    """``[QSG_RHI_BACKEND, KMP_BLOCKTIME]`` after ``import main`` in a fresh process, by the
    ``KMP_BLOCKTIME`` the user set (None: none). The processes run at the same time."""
    started = {
        blocktime: _import_main(
            tmp_path_factory.mktemp("import_main"),
            **({} if blocktime is None else {"KMP_BLOCKTIME": blocktime}),
        )
        for blocktime in (None, "7")
    }
    return {blocktime: _printed(proc) for blocktime, proc in started.items()}


@pytest.fixture(scope="module")
def env_after_importing_main(imports_of_main):
    """``[QSG_RHI_BACKEND, KMP_BLOCKTIME]`` after ``import main`` in a fresh process."""
    return imports_of_main[None]


def test_importing_main_sets_d3d11_only_on_windows(env_after_importing_main):
    assert env_after_importing_main[0] == ("d3d11" if sys.platform == "win32" else "None")


def test_importing_main_lets_idle_openmp_workers_sleep(env_after_importing_main):
    """KMP_BLOCKTIME=0 unless the user set one (P3-01); ``import main`` imports no torch
    (``test_startup_import_budget.py``), so it is set before anything can."""
    assert env_after_importing_main[1] == "0"


def test_a_blocktime_the_user_set_wins(imports_of_main):
    """``main`` only defaults KMP_BLOCKTIME: a value the user set stays (P3-01)."""
    assert imports_of_main["7"][1] == "7"


# --- main() ------------------------------------------------------------------------

_MAIN_DRIVER = """
import sys
from pathlib import Path
sys.path.insert(0, {repo!r})
import main
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication, QMainWindow, QMessageBox

class StubWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        app = QApplication.instance()  # inspected while main() is running
        print("QSS", len(app.styleSheet()) > 1000)
        print("ICON", not app.windowIcon().isNull())
        print("NAMES", app.organizationName(), app.applicationName())
        print("HOOK", sys.excepthook.__qualname__)
        if {fail!r}:
            raise RuntimeError("checkpoint missing")
        QTimer.singleShot(100, app.quit)

main.MainWindow = StubWindow
QMessageBox.critical = staticmethod(lambda *a, **k: print("CRITICAL-BOX", a[2]))
print("RC", main.main(["sprouts"]))
"""


def test_main_starts_from_any_working_directory(tmp_path):
    log_dir = tmp_path / "logs"
    proc = _run(
        _MAIN_DRIVER.format(repo=str(REPO_ROOT), fail=False),
        cwd=tmp_path,  # not the repository: the QSS and icon must still resolve
        env_extra={"SPROUTS_LOG_DIR": str(log_dir)},
    )
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout.splitlines()
    assert "RC 0" in out
    assert "QSS True" in out
    assert "ICON True" in out
    assert "NAMES LeakeyLab SPROUTS" in out
    assert "HOOK ExceptionHooks._sys_hook" in out
    assert (log_dir / "app.log").is_file()


def test_main_reports_a_startup_failure_instead_of_vanishing(tmp_path):
    log_dir = tmp_path / "logs"
    proc = _run(
        _MAIN_DRIVER.format(repo=str(REPO_ROOT), fail=True),
        cwd=tmp_path,
        env_extra={"SPROUTS_LOG_DIR": str(log_dir)},
    )
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout.splitlines()
    assert "RC 1" in out
    assert any(line.startswith("CRITICAL-BOX") and "checkpoint missing" in line for line in out)
    log_text = (log_dir / "app.log").read_text(encoding="utf-8")
    assert "RuntimeError: checkpoint missing" in log_text


def test_main_runs_without_a_log_file(tmp_path):
    """The log folder cannot be made (a file is in the way): ``main()`` still gets as far as
    the window, with no crash log, and its failure box names no log file."""
    in_the_way = tmp_path / "logs"
    in_the_way.write_text("a file where the log folder should be", encoding="utf-8")
    proc = _run(
        _MAIN_DRIVER.format(repo=str(REPO_ROOT), fail=True),
        cwd=tmp_path,
        env_extra={"SPROUTS_LOG_DIR": str(in_the_way)},
    )
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout.splitlines()
    assert "RC 1" in out
    (box,) = [line for line in out if line.startswith("CRITICAL-BOX")]
    assert "checkpoint missing" in box
    assert "Details were written" not in proc.stdout  # the box's text, all its lines
    assert in_the_way.is_file()  # nothing was written in its place


def test_an_import_failure_is_written_to_the_log(tmp_path):
    log_dir = tmp_path / "logs"
    proc = _run(
        f"""
        import sys
        sys.path.insert(0, {str(REPO_ROOT)!r})
        sys.modules["app.gui.main_window"] = None  # simulate a broken app module
        import main
        """,
        cwd=tmp_path,
        env_extra={"SPROUTS_LOG_DIR": str(log_dir)},
    )
    assert proc.returncode != 0  # still fatal: there is nothing to run
    log_text = (log_dir / "app.log").read_text(encoding="utf-8")
    assert "Could not import the application" in log_text
    assert "import of app.gui.main_window halted" in log_text  # the traceback's last line


def test_main_keeps_the_names_the_ui_harness_uses(tmp_path):
    names = ["resource_path", "apply_stylesheet", "MainWindow", "main"]
    proc = _run(
        f"""
        import sys
        sys.path.insert(0, {str(REPO_ROOT)!r})
        import main
        print(*[callable(getattr(main, name, None)) for name in {names!r}])
        """,
        cwd=tmp_path,
        env_extra={"SPROUTS_LOG_DIR": str(tmp_path / "logs")},
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().splitlines()[-1] == "True True True True"


# --- main.main teardown --------------------------------------------------------------

_TEARDOWN_DRIVER = """
import sys
sys.path.insert(0, {repo!r})
import main
from PyQt6 import sip
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication, QMainWindow

class StubWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        QTimer.singleShot(50, self.close)  # the user closes the window

main.MainWindow = StubWindow
rc = main.main(["sprouts"])
app, window = main._session
print("RC", rc)
print("APP ALIVE", QApplication.instance() is app and not sip.isdeleted(app))
print("WINDOW DELETED", sip.isdeleted(window))
"""


def _run_main(code: str, tmp_path, timeout: int = 120) -> subprocess.CompletedProcess:
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", PYTHONDONTWRITEBYTECODE="1")
    env.update(
        SPROUTS_LOG_DIR=str(tmp_path / "logs"),
        SPROUTS_CONFIG_DIR=str(tmp_path / "cfg"),
        ROOT_VIEWER_ENABLE_OPENGL_VIEWPORT="0",
    )
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def test_main_deletes_the_window_while_the_application_still_exists(tmp_path):
    proc = _run_main(_TEARDOWN_DRIVER.format(repo=str(REPO_ROOT)), tmp_path)

    assert proc.returncode == 0, proc.stderr
    out = proc.stdout.splitlines()
    assert "RC 0" in out
    assert "APP ALIVE True" in out
    assert "WINDOW DELETED True" in out


_EXIT_DRIVER = """
import sys
sys.path.insert(0, {repo!r})
from tests.correct_goldens_support import make_synth_project
folder, image = make_synth_project({root!r}, "synth_model_mask_and_skeleton")
import main
from PyQt6.QtCore import QTimer

class Window(main.MainWindow):
    def __init__(self):
        super().__init__()
        QTimer.singleShot(0, self._open_then_close)

    def _open_then_close(self):
        self.correct_editor.load_image(image, folder)
        self.toggle_correct()
        QTimer.singleShot(30, self.close)

main.MainWindow = Window
sys.exit(main.main(["sprouts"]))
"""


@pytest.mark.skipif(
    os.environ.get("SPROUTS_PERF_TESTS") != "1",
    reason="about 2 s per fresh app; opt-in with SPROUTS_PERF_TESTS=1",
)
def test_closing_the_app_with_a_correct_page_open_never_crashes_at_exit(tmp_path):
    runs = int(os.environ.get("SPROUTS_EXIT_RUNS", "40"))
    codes = []
    for index in range(runs):
        root = tmp_path / f"run{index}"
        root.mkdir()
        code = _EXIT_DRIVER.format(repo=str(REPO_ROOT), root=str(root))
        proc = _run_main(code, root)
        codes.append(proc.returncode)
        assert "Traceback" not in proc.stderr, proc.stderr
    assert codes == [0] * runs  # 0xC0000005 is 3221225477


# --- the palette ---------------------------------------------------------------------
def test_disabled_text_uses_the_disabled_tier(app):
    import main
    from app.gui.widgets import tokens

    palette = main._sprouts_palette()
    group = QPalette.ColorGroup.Disabled
    for role in (QPalette.ColorRole.Text, QPalette.ColorRole.WindowText):
        assert palette.color(group, role).name() == tokens.TEXT_DISABLED
