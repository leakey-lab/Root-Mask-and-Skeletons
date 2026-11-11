"""The suite's own set-up (``tests/conftest.py``).

- One QApplication for the session, as in a full run (OpenGL contexts shared, raster
  canvases by default); a module's narrower ``app`` fixture still lets a test ask for
  ``qapp``.
- The user's config folder is out of reach: every test has a temp folder of its own, so one
  test's saved defaults (the Enhance defaults) never reach a later test.
- The suite logs into a temp folder, never into the user's application log (X5).
- The keeper census: a test that leaves a kept thread running gets a warning naming it,
  once, and never a failure. Run in a child pytest with the conftest loaded as a plugin,
  over a small test file of three tests: the first starts a thread and leaves it running,
  the second runs while it still runs, the third lets it end.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import replace
from pathlib import Path

import pytest
from PyQt6.QtCore import QCoreApplication, Qt
from PyQt6.QtWidgets import QApplication

from app import logging_config
from app.enhancement import EnhanceDefaultsStore
from app.user_paths import user_config_dir

ROOT = Path(__file__).resolve().parents[1]

# ================================================================== the one QApplication


def test_app_and_qapp_are_the_one_application(app, qapp):
    assert isinstance(app, QApplication)
    assert app is qapp is QApplication.instance()


def test_the_application_shares_opengl_contexts(app):
    assert QCoreApplication.testAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)


class TestAModuleWithItsOwnApp:
    """A module's ``app`` fixture of a narrower scope (a themed app, say) still lets a test
    ask for the session's ``qapp``: no scope error."""

    @pytest.fixture(scope="class")
    def app(self):
        return QApplication.instance() or QApplication([])

    def test_qapp_is_the_same_application(self, app, qapp):
        assert app is qapp


# ======================================================== the per-test config folder

_FOLDERS: list[str] = []  # the config folders of the tests below that ran
_SAVED: list[str] = []  # the defaults file a test below saved


def _in_temp(path: str) -> bool:
    return Path(path).resolve().is_relative_to(Path(tempfile.gettempdir()).resolve())


@pytest.fixture(scope="module")
def module_config_dir():
    """``SPROUTS_CONFIG_DIR`` as a module fixture sees it: the session's folder."""
    return os.environ["SPROUTS_CONFIG_DIR"]


def test_the_suite_keeps_user_config_in_a_temp_folder():
    configured = os.environ["SPROUTS_CONFIG_DIR"]
    assert _in_temp(configured)
    assert user_config_dir(ensure=False) == configured
    assert Path(EnhanceDefaultsStore().user_path).parent == Path(configured)


def test_module_fixtures_are_kept_out_of_the_user_folder_too(module_config_dir):
    assert _in_temp(module_config_dir)
    configured = os.environ["SPROUTS_CONFIG_DIR"]
    assert configured != module_config_dir  # this test's own folder ...
    assert Path(configured).is_relative_to(Path(module_config_dir))  # ... in the session's


@pytest.mark.parametrize("run", [1, 2])
def test_every_test_has_its_own_config_folder(run):
    configured = os.environ["SPROUTS_CONFIG_DIR"]
    assert configured not in _FOLDERS
    assert not os.path.exists(configured)  # made when the app first writes there
    _FOLDERS.append(configured)


def test_defaults_a_test_saves_stay_in_its_folder():
    store = EnhanceDefaultsStore()
    assert store.save(replace(store.load(), tile=3))
    assert EnhanceDefaultsStore().load().tile == 3
    _SAVED.append(store.user_path)


def test_the_next_test_starts_from_the_factory_defaults():
    if not _SAVED:
        pytest.skip("runs after test_defaults_a_test_saves_stay_in_its_folder")
    assert os.path.exists(_SAVED[0])
    store = EnhanceDefaultsStore()
    assert store.user_path != _SAVED[0]
    assert not os.path.exists(store.user_path)
    assert store.load().tile != 3


# ================================================================== the temp log folder


def test_the_suite_logs_into_a_temp_folder(monkeypatch):
    configured = os.environ["SPROUTS_LOG_DIR"]
    assert logging_config.default_log_dir() == Path(configured)
    assert Path(configured).resolve().is_relative_to(Path(tempfile.gettempdir()).resolve())

    from app.user_paths import user_log_dir

    monkeypatch.delenv("SPROUTS_LOG_DIR")
    assert Path(user_log_dir()) != Path(configured)


# ===================================================================== the keeper census

CHILD_TESTS = textwrap.dedent(
    """
    import threading

    from PyQt6.QtCore import QCoreApplication

    from app.workers import GuardedThread, kept_threads

    release = threading.Event()


    class Held(GuardedThread):
        def work(self):
            release.wait(30)


    def test_leaves_a_job_running(qapp):
        thread = Held()
        thread.setObjectName("held-job")
        thread.start()


    def test_runs_while_it_still_runs(qapp):
        assert any(t.objectName() == "held-job" and t.isRunning() for t in kept_threads())


    def test_lets_it_end(qapp):
        release.set()
        for thread in kept_threads():
            assert thread.wait(10_000)
        QCoreApplication.processEvents()
    """
)


def test_a_kept_thread_left_running_is_named_once_and_fails_nothing(tmp_path):
    (tmp_path / "test_child.py").write_text(CHILD_TESTS, encoding="utf-8")
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "tests.conftest", "-p", "no:cacheprovider"]
        + ["-p", "no:dash", "-o", "addopts="]  # pyproject's addopts minus -q (see there)
        + ["--rootdir", str(tmp_path), str(tmp_path / "test_child.py")],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    out = result.stdout
    assert result.returncode == 0, out[-3000:] + result.stderr[-2000:]
    assert "3 passed" in out, out[-3000:]
    assert out.count("KeptThreadRunning: still running after") == 1, out[-3000:]
    assert "test_child.py::test_leaves_a_job_running: held-job" in out, out[-3000:]
