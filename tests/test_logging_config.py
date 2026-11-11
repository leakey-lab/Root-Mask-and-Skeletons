"""Logging set-up (``app/logging_config.py``).

``setup_logging`` configures the root logger once (stderr + a rotating ``app.log`` whose
first line is a version banner), in the folder ``SPROUTS_LOG_DIR`` names, else the per-user
log folder. It survives a process without stderr (``pythonw.exe``) and a folder that
cannot be made: then only stderr logs, and the start goes on.

Logging is process-global state, so every test runs in a fresh interpreter with the log
folder in ``tmp_path``: nothing is written outside it.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

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


def test_setup_logging_writes_a_version_banner_and_is_idempotent(tmp_path):
    proc = _run(
        f"""
        import logging
        from pathlib import Path
        from app.logging_config import current_log_file, setup_logging
        first = setup_logging(Path({str(tmp_path)!r}))
        count = len(logging.getLogger().handlers)
        second = setup_logging(Path({str(tmp_path)!r}) / "elsewhere")
        print(first == second == current_log_file(), count == len(logging.getLogger().handlers))
        logging.getLogger("probe").info("hello log")
        """,
        cwd=REPO_ROOT,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.split() == ["True", "True"]
    text = (tmp_path / "app.log").read_text(encoding="utf-8")
    assert "SPROUTS" in text and "Python" in text
    assert "hello log" in text


def test_log_dir_env_override_wins_over_user_paths(tmp_path):
    env_dir = tmp_path / "from-env"
    proc = _run(
        """
        from app.logging_config import setup_logging
        print(setup_logging())
        """,
        cwd=REPO_ROOT,
        env_extra={"SPROUTS_LOG_DIR": str(env_dir)},
    )
    assert proc.returncode == 0, proc.stderr
    assert Path(proc.stdout.strip()) == env_dir / "app.log"


def test_default_log_dir_comes_from_user_paths_when_available(tmp_path):
    user_dir = tmp_path / "user-logs"
    proc = _run(
        f"""
        import sys, types
        from pathlib import Path
        fake = types.ModuleType("app.user_paths")
        fake.user_log_dir = lambda: Path({str(user_dir)!r})
        sys.modules["app.user_paths"] = fake
        from app.logging_config import setup_logging
        print(setup_logging())
        """,
        cwd=REPO_ROOT,
    )
    assert proc.returncode == 0, proc.stderr
    assert Path(proc.stdout.strip()) == user_dir / "app.log"


def test_a_log_folder_that_cannot_be_made_leaves_stderr_logging(tmp_path):
    """A file where the log folder should be: no file log, a warning, and the start goes on
    (``main.py`` calls this at import, before any window)."""
    in_the_way = tmp_path / "logs"
    in_the_way.write_text("a file where the log folder should be", encoding="utf-8")
    proc = _run(
        f"""
        import logging
        from pathlib import Path
        from app.logging_config import current_log_file, setup_logging
        print(setup_logging(Path({str(in_the_way)!r})), current_log_file())
        logging.getLogger("probe").warning("still on stderr")
        """,
        cwd=REPO_ROOT,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.split() == ["None", "None"]
    assert "File logging disabled" in proc.stderr
    assert "still on stderr" in proc.stderr


def test_setup_logging_survives_a_missing_stderr(tmp_path):
    proc = _run(
        f"""
        import logging, sys
        from pathlib import Path
        from app.logging_config import setup_logging
        real_stdout = sys.stdout
        sys.stderr = None  # pythonw.exe / windowed PyInstaller builds
        setup_logging(Path({str(tmp_path)!r}))
        logging.getLogger("probe").warning("no console here")
        real_stdout.write("ok")
        """,
        cwd=REPO_ROOT,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == "ok"
    assert "no console here" in (tmp_path / "app.log").read_text(encoding="utf-8")
