"""Background jobs report an outcome whatever ends them (KNOWN_ISSUES #7).

A ``BaseException`` that is not an ``Exception`` (``SystemExit``, ``KeyboardInterrupt``, or
any other) used to end a job's thread without a word, so the GUI waited for good:

- the Measure CSV parse (``MetricsController.prewarm``) left ``metrics.loading`` True:
  no other parse, the warm-ups blocked and the bar never refreshed;
- a folder scan (``ImageLoaderThread``) caught ``OSError`` only: anything else reported
  neither outcome and the loading card stayed up.

Each now logs the traceback and reports a failure (never re-raised: the thread has no one
to hand it to, and PyQt aborts the process on an exception that escapes a pool job or
``QThread.run`` under the default ``sys.excepthook``). The tests record ``sys.excepthook``:
nothing may reach it. The same promise for the start-up device check is in
``test_device_check.py``, for the warm-ups in ``test_warmup_queue.py`` and for
``GuardedThread`` in ``test_workers_guarded_thread.py``.
"""

from __future__ import annotations

import os
import time
from types import SimpleNamespace

import pytest
from PyQt6.QtCore import QCoreApplication, QEvent, QThreadPool
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from app.gui import image_manager as im
from app.gui.window import project_loader
from app.gui.window.metrics import MetricsController
from tests.qt_teardown import flush_deleted


class Abort(BaseException):
    """Neither an ``Exception`` nor ``SystemExit`` / ``KeyboardInterrupt``."""


ENDINGS = [SystemExit(3), KeyboardInterrupt(), Abort("stopped")]
ENDING_IDS = ["exit", "interrupt", "other"]


@pytest.fixture(autouse=True)
def nothing_escapes(app, slot_errors):
    """No exception may leave a job's thread (PyQt hands it to ``sys.excepthook``):
    ``slot_errors`` asserts none once the pool's jobs have ended and their objects are gone."""
    yield
    assert QThreadPool.globalInstance().waitForDone(10_000)
    flush_deleted()


def wait(predicate, timeout: float = 5.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        QApplication.processEvents()
        if predicate():
            return True
        QTest.qWait(10)
    return predicate()


def raiser(ending: BaseException):
    def raise_it(*_args, **_kwargs):
        raise ending

    return raise_it


# ------------------------------------------------------------------ the Measure CSV parse
class Index:
    """A measurement index whose first parse ends with ``ending``; later ones succeed."""

    def __init__(self, ending: BaseException) -> None:
        self.ending = ending
        self.prewarmed = 0

    def needs_load(self) -> bool:
        return True

    def prewarm(self) -> None:
        self.prewarmed += 1
        if self.prewarmed == 1:
            raise self.ending


def metrics_window() -> SimpleNamespace:
    return SimpleNamespace(
        image_manager=SimpleNamespace(original_folder=None, images={}),
        display_controller=SimpleNamespace(current_name=None),
        stage_status=SimpleNamespace(check=lambda *a, **k: None),
        _current_stage=lambda: None,
    )


@pytest.mark.parametrize("ending", ENDINGS, ids=ENDING_IDS)
def test_a_parse_ended_by_a_base_exception_still_ends_the_loading(ending, caplog):
    controller = MetricsController(metrics_window())
    controller.signals.ready.connect(controller.on_ready)  # as the window connects it
    controller.index = index = Index(ending)

    controller.prewarm()
    assert controller.loading

    assert wait(lambda: not controller.loading), "the metrics stayed loading for good"
    assert "Could not read the measurement CSVs" in caplog.text
    controller.prewarm()  # the next parse is not blocked
    assert wait(lambda: index.prewarmed == 2 and not controller.loading)


# ------------------------------------------------------------------ the folder scan
def loader_window(calls: list) -> SimpleNamespace:
    """A window the project loader reports to, with a real :class:`ImageManager`."""
    window = SimpleNamespace(
        _loading_project=True,
        loading_overlay=SimpleNamespace(
            hide=lambda: calls.append("card hidden"), set_progress=lambda value: None
        ),
        empty_state=SimpleNamespace(set_loading=lambda name: None),
        task_progress=SimpleNamespace(
            set_progress=lambda value: None,
            fail=lambda message: calls.append(("failed", message)),
            finish=lambda message: calls.append(("finished", message)),
        ),
        notify=lambda message, kind="success", timeout=3200, key=None: calls.append(
            (kind, message)
        ),
        app_stack=SimpleNamespace(setCurrentIndex=lambda i: calls.append(("page", i))),
    )
    window.update_loading_progress = lambda v: project_loader.update_loading_progress(window, v)
    window.on_loading_failed = lambda f, m: project_loader.on_loading_failed(window, f, m)
    window.on_loading_empty = lambda f: project_loader.on_loading_empty(window, f)
    window.on_loading_finished = lambda *args: calls.append("loaded")
    window._end_failed_load = lambda m, k: project_loader.end_failed_load(window, m, k)
    window.image_manager = im.ImageManager(window)
    return window


@pytest.mark.parametrize(
    ("where", "ending"),
    [
        ("index_names", ValueError("a name the index cannot place")),
        ("index_names", MemoryError()),
        *(("scan_image_folder", ending) for ending in ENDINGS),
    ],
    ids=["value-error", "memory-error", *ENDING_IDS],
)
def test_a_scan_ended_by_anything_but_an_oserror_reports_a_failure(
    tmp_path, monkeypatch, caplog, where, ending
):
    if where == "index_names":
        monkeypatch.setattr(im, "index_names", raiser(ending))
    else:
        monkeypatch.setattr(im.ProjectLayout, "scan_image_folder", raiser(ending))
    folder = str(tmp_path / "trial")
    os.mkdir(folder)
    calls: list = []
    window = loader_window(calls)

    window.image_manager.load_images(folder)

    assert wait(lambda: "card hidden" in calls), "the loading card stayed up"
    assert window._loading_project is False and "loaded" not in calls
    (message,) = [entry[1] for entry in calls if entry[0] == "failed"]
    assert folder in message and type(ending).__name__ in message
    assert ("danger", message) in calls and ("page", 0) in calls  # nothing loaded before
    assert window.image_manager.images == {}
    assert "Failed to scan image folder" in caplog.text
    assert wait(lambda: window.image_manager.loader_thread is None)  # the thread is let go


def test_an_unreadable_folder_keeps_its_message(tmp_path, monkeypatch):
    """An ``OSError`` still reads "Cannot read <folder>: <why>"."""
    monkeypatch.setattr(
        im.ProjectLayout, "scan_image_folder", raiser(PermissionError(13, "Access is denied"))
    )
    folder = str(tmp_path / "trial")
    os.mkdir(folder)
    calls: list = []
    window = loader_window(calls)

    window.image_manager.load_images(folder)

    assert wait(lambda: "card hidden" in calls)
    assert ("failed", f"Cannot read {folder}: Access is denied") in calls
