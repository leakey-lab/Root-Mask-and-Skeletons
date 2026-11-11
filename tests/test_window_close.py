"""Closing the window (``app/gui/window/lifecycle.py``).

Promises: an editor that keeps unsaved edits, or a job the user keeps running, cancels the
close and touches nothing else; otherwise the job registry is shut down first (a job chained
to the stopped one never starts), a job that outlives the first wait gets the window hidden
and a second wait, the warm-ups and the device check are dropped, the dashboards are closed
(a failure is only logged) and the user's layout is saved (not the one collapsed for a
dashboard); the next window opens with that layout.

The first tests run the part against a stand-in window (each branch, the order of the
steps); the last ones close a real window.
"""

from __future__ import annotations

import logging
import os
from types import SimpleNamespace

from PyQt6.QtWidgets import QMessageBox

from app.gui.panels import Panel
from app.gui.window import lifecycle
from tests.window_support import make_window as make_window  # fixture
from tests.window_support import mw as mw  # fixture
from tests.window_support import no_delayed_temp_sweep as no_delayed_temp_sweep  # fixture
from tests.window_support import open_folder, write_photos
from tests.window_support import settings as settings  # fixture

STEMS = [f"24EFCAM1_T001_L0{d}_2024.07.18_1115{d}_001_SMM" for d in (11, 12, 13)]


class Event:
    """A close event: records ``ignore()``."""

    def __init__(self) -> None:
        self.ignored = False

    def ignore(self) -> None:
        self.ignored = True


def closing_window(calls: list, *, keeps_edits=False, busy=False, stops=(True,), viz_fails=False):
    """A window to close: one editor, a job registry, the clean-up steps (all recorded)."""
    results = list(stops)
    editor = SimpleNamespace(maybe_save_changes=lambda: calls.append("prompt") or not keeps_edits)

    def close_length():
        calls.append("close length")
        if viz_fails:
            raise RuntimeError("the page is gone")

    window = SimpleNamespace(
        jobs=SimpleNamespace(
            is_busy=lambda: busy,
            running_name=lambda: "Generating masks",
            shutdown=lambda: calls.append("shutdown"),
            stop_all=lambda timeout_ms: calls.append(("stop_all", timeout_ms)) or results.pop(0),
        ),
        feedback=SimpleNamespace(status=lambda message, _timeout=0: calls.append(message)),
        hide=lambda: calls.append("hide"),
        warmups=SimpleNamespace(shutdown=lambda: calls.append("warmups")),
        device_check=SimpleNamespace(cancel=lambda: calls.append("device check")),
        root_length_viz=object(),
        root_area_viz=None,
        close_root_length_visualization=close_length,
        close_root_area_visualization=lambda: calls.append("close area"),
        _save_window_state=lambda: calls.append("save state"),
    )
    window._editors = lambda: {Panel.CORRECT: editor}
    window._stop_jobs_for_close = lambda: lifecycle.stop_jobs_for_close(window)
    return window


def answer(monkeypatch, button) -> None:
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: button))


def test_an_editor_that_keeps_its_edits_cancels_the_close(app):
    calls, event = [], Event()
    window = closing_window(calls, keeps_edits=True)
    assert lifecycle.close_event(window, event) is False
    assert event.ignored and calls == ["prompt"]  # no job was touched


def test_a_job_the_user_keeps_running_cancels_the_close(app, monkeypatch):
    answer(monkeypatch, QMessageBox.StandardButton.No)
    calls, event = [], Event()
    window = closing_window(calls, busy=True)
    assert lifecycle.close_event(window, event) is False
    assert event.ignored and calls == ["prompt"]  # the registry still takes jobs


def test_a_job_that_outlives_both_waits_is_logged_and_the_close_goes_on(app, monkeypatch, caplog):
    answer(monkeypatch, QMessageBox.StandardButton.Yes)
    calls, event = [], Event()
    window = closing_window(calls, busy=True, stops=(False, False))
    with caplog.at_level(logging.WARNING, logger=lifecycle.logger.name):
        assert lifecycle.close_event(window, event) is True
    assert not event.ignored
    assert calls == [
        "prompt",
        "shutdown",
        "Stopping: Generating masks…",
        ("stop_all", lifecycle._CLOSE_WAIT_MS),
        "hide",
        ("stop_all", lifecycle._CLOSE_EXTRA_WAIT_MS),
        "warmups",
        "device check",
        "close length",
        "save state",
    ]
    assert "did not stop in time" in caplog.text


def test_a_failing_dashboard_clean_up_is_logged_and_the_layout_still_saved(app, caplog):
    calls, event = [], Event()
    window = closing_window(calls, viz_fails=True)
    with caplog.at_level(logging.ERROR, logger=lifecycle.logger.name):
        assert lifecycle.close_event(window, event) is True
    assert calls[-2:] == ["close length", "save state"]
    assert "Error cleaning up visualizations on close" in caplog.text


def test_the_saved_layout_is_the_users_not_the_collapsed_one(caplog):
    calls = []
    window = SimpleNamespace(
        _persist_window_state=True,
        _pre_viz_splitter_sizes=[300, 900],
        left_panel=SimpleNamespace(setVisible=lambda shown: calls.append(("library", shown))),
        body_splitter=SimpleNamespace(setSizes=lambda sizes: calls.append(("sizes", sizes))),
        window_state=SimpleNamespace(save=lambda w, splitter: calls.append("save")),
    )
    lifecycle.save_window_state(window)
    assert calls == [("library", True), ("sizes", [300, 900]), "save"]

    window._persist_window_state = False
    lifecycle.save_window_state(window)
    assert calls[3:] == []  # a window that does not persist its layout saves nothing

    def fail(*_a):
        raise OSError("read-only settings")

    window._persist_window_state, window.window_state = True, SimpleNamespace(save=fail)
    with caplog.at_level(logging.ERROR, logger=lifecycle.logger.name):
        lifecycle.save_window_state(window)  # logged, never raised
    assert "Failed to save the window state" in caplog.text


def test_close_is_cancelled_by_an_editor_that_keeps_its_edits(mw, app):
    mw.show()
    mw.correct_editor.maybe_save_changes = lambda: False
    mw.close()
    app.processEvents()
    assert mw.isVisible()


def test_close_while_a_job_runs_asks_first(mw, app, monkeypatch):
    mw.show()
    stopped = []
    monkeypatch.setattr(mw.jobs, "is_busy", lambda: True)
    monkeypatch.setattr(mw.jobs, "stop_all", lambda timeout_ms=5000: stopped.append(1) or True)
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.No)
    mw.close()
    app.processEvents()
    assert mw.isVisible() and not stopped
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    mw.close()
    assert stopped == [1]


def test_window_state_and_last_folder_round_trip(make_window, settings, tmp_path):
    folder = write_photos(tmp_path / "proj", STEMS)
    first = make_window()
    first.resize(790, 700)
    open_folder(first, folder, len(STEMS))
    first.close()
    assert settings.value("window/geometry") is not None
    assert os.path.normpath(first.window_state.last_folder()) == os.path.normpath(str(folder))

    second = make_window()
    assert (second.width(), second.height()) == (790, 700)


def test_an_idle_close_shuts_the_job_registry_first(app):
    """No job runs: the registry is still shut before anything else is cleaned up, so a job
    chained to one that just ended (Generate: masks, then skeletons) cannot start."""
    calls, event = [], Event()
    window = closing_window(calls)
    assert lifecycle.close_event(window, event) is True
    assert calls == ["prompt", "shutdown", "warmups", "device check", "close length", "save state"]
