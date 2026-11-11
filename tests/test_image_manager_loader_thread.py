"""The Library's folder scan (``ImageLoaderThread``) on the one worker base of the app.

It is a ``GuardedThread``: the error boundary of ``app.workers`` reports anything that ends
a scan (with the messages the scan always had, naming the folder), and the thread keeper
holds a scan's thread from ``start()`` until Qt's own ``finished``, so a superseded scan
needs no bookkeeping of the manager. A scan that is cancelled (another folder was opened)
still emits nothing.

A scan's thread is freed as soon as it has ended, without the cyclic GC: no slot holds the
thread (a ``partial`` holding it made a reference cycle that only the cyclic collector
broke, never once ``gc.freeze()`` had frozen it: 17.2 MB of a 46,178-image scan to free in a
41-51 ms pause), and the manager lets it go when it ends (delivered, failed or superseded).
Those tests turn the automatic GC off and watch the thread through a weak reference.
"""

from __future__ import annotations

import logging
import os
import threading
import time
import weakref

import pytest
from PyQt6.QtCore import QCoreApplication, QEvent
from PyQt6.QtWidgets import QApplication

from app.gui import image_manager as im
from app.workers import LOGGER_NAME, kept_threads
from tests.leak_census import assert_freed, no_gc_fixture  # noqa: F401
from tests.qt_wait import wait_until
from tests.window_support import write_photos

A_STEMS = ["plot_T1_L1_2024.01.01_120000", "plot_T1_L2_2024.01.01_120000"]
B_STEMS = ["plot_T2_L1_2024.01.01_120000"]

pytestmark = pytest.mark.usefixtures("no_temp_sweep")


def _project(folder, stems) -> str:
    return str(write_photos(folder, stems, size=(8, 6), suffix=".png"))


def _record(thread) -> dict[str, list]:
    got: dict[str, list] = {"progress": [], "finished": [], "failed": [], "cancelled": []}
    thread.progress.connect(got["progress"].append)
    thread.finished.connect(lambda *args: got["finished"].append(args))
    thread.failed.connect(got["failed"].append)
    thread.cancelled.connect(lambda: got["cancelled"].append(True))
    return got


@pytest.fixture
def held_scan(monkeypatch):
    """The scan of the folder ``hold[0]`` waits inside ``scan_image_folder`` until
    ``release`` is set (set on teardown too). Returns ``(hold, held, release)``."""
    hold, held, release = [None], threading.Event(), threading.Event()
    real_scan = im.ProjectLayout.scan_image_folder

    def scan_image_folder(layout):
        if hold[0] and os.path.normcase(layout.root) == os.path.normcase(hold[0]):
            held.set()
            release.wait(10)
        return real_scan(layout)

    monkeypatch.setattr(im.ProjectLayout, "scan_image_folder", scan_image_folder)
    yield hold, held, release
    release.set()


@pytest.mark.parametrize(
    ("exc", "message"),
    [
        (PermissionError(13, "Access is denied"), "Cannot read {folder}: Access is denied"),
        (OSError("no listing"), "Cannot read {folder}: no listing"),
        (ValueError("a name"), "Could not scan {folder}: ValueError: a name"),
        (SystemExit(3), "Could not scan {folder}: SystemExit: 3"),
    ],
    ids=["os-error", "os-error-without-strerror", "error", "exit"],
)
def test_a_failed_scan_reports_its_message_and_names_the_folder_in_the_log(
    app, tmp_path, monkeypatch, caplog, exc, message
):
    def fail(_layout):
        raise exc

    monkeypatch.setattr(im.ProjectLayout, "scan_image_folder", fail)
    folder = str(tmp_path)
    thread = im.ImageLoaderThread(folder)
    got = _record(thread)

    with caplog.at_level(logging.ERROR):
        thread.run()  # the thread's body, on this thread

    assert got["failed"] == [message.format(folder=folder)]
    assert got["progress"] == got["finished"] == got["cancelled"] == []
    named = [r for r in caplog.records if r.name == im.__name__]
    assert [r.getMessage() for r in named] == [f"Failed to scan image folder {folder}"]
    assert any(r.exc_info for r in caplog.records if r.name == LOGGER_NAME)  # the traceback


def test_a_missing_folder_is_reported_without_a_traceback(app, tmp_path, caplog):
    folder = str(tmp_path / "gone")
    thread = im.ImageLoaderThread(folder)
    got = _record(thread)

    with caplog.at_level(logging.WARNING):
        thread.run()

    assert got["failed"] == [f"Folder not found: {folder}"] and got["finished"] == []
    assert caplog.records == []


def test_a_cancelled_scan_emits_nothing(app, tmp_path, held_scan):
    hold, held, release = held_scan
    hold[0] = folder = _project(tmp_path / "a", A_STEMS)
    thread = im.ImageLoaderThread(folder)
    got = _record(thread)
    thread.start()
    assert held.wait(10)

    thread.cancel()
    release.set()

    assert thread.wait(10000)
    assert wait_until(lambda: thread not in kept_threads())
    assert got == {"progress": [], "finished": [], "failed": [], "cancelled": []}


def test_a_superseded_scan_is_kept_by_the_thread_keeper_until_it_ends(app, tmp_path, held_scan):
    hold, held, release = held_scan
    a, b = _project(tmp_path / "a", A_STEMS), _project(tmp_path / "b", B_STEMS)
    hold[0] = a
    manager = im.ImageManager(None)
    manager.load_images(a)
    first = manager.loader_thread
    assert held.wait(10)

    manager.load_images(b)  # the first scan is asked to stop and let go by the manager
    second = manager.loader_thread
    try:
        assert second is not first and first.isRunning() and first.is_cancelled()
        assert first in kept_threads() and second in kept_threads()
    finally:
        release.set()

    assert wait_until(lambda: manager.loader_thread is None)
    assert sorted(manager.images) == sorted(B_STEMS)  # the superseded scan delivered nothing
    assert first.wait(10000)
    assert wait_until(lambda: first not in kept_threads() and second not in kept_threads())


# ------------------------------------------------------------------ freed at once

WAIT_S = 10.0


def _pump(until, timeout: float = WAIT_S) -> bool:
    """Run the event loop (queued slots, deferred deletes) until ``until()`` or timeout."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        QApplication.processEvents()
        if until():
            return True
        time.sleep(0.005)
    return until()


def test_a_finished_scan_frees_its_thread(app, tmp_path, no_gc):
    manager = im.ImageManager(None)
    manager.load_images(_project(tmp_path / "a", A_STEMS))
    thread = weakref.ref(manager.loader_thread)

    assert_freed(thread, "the finished loader thread", pump=True, timeout=WAIT_S)
    assert sorted(manager.images) == sorted(A_STEMS)  # it delivered first
    assert manager.skeleton_stems == set() and manager.get_leaf_labels()
    assert manager.loader_thread is None


def test_a_failed_scan_frees_its_thread(app, tmp_path, no_gc):
    manager = im.ImageManager(None)
    manager.load_images(str(tmp_path / "gone"))
    thread = weakref.ref(manager.loader_thread)

    assert_freed(thread, "the failed loader thread", pump=True, timeout=WAIT_S)
    assert manager.images == {} and manager.loader_thread is None


def test_a_superseded_scan_frees_its_thread_when_it_ends(app, tmp_path, monkeypatch, no_gc):
    """Opening another folder while a scan runs never kills it (PG-15): the old scan is
    kept until its thread ends, then freed; the new one delivers and is freed too."""
    a, b = _project(tmp_path / "a", A_STEMS), _project(tmp_path / "b", B_STEMS)
    held, release = threading.Event(), threading.Event()
    real_scan = im.ProjectLayout.scan_image_folder

    def scan_image_folder(layout):
        if os.path.normcase(layout.root) == os.path.normcase(a):
            held.set()
            release.wait(10)  # the first scan is still running when b is opened
        return real_scan(layout)

    monkeypatch.setattr(im.ProjectLayout, "scan_image_folder", scan_image_folder)
    manager = im.ImageManager(None)
    manager.load_images(a)
    first = weakref.ref(manager.loader_thread)
    assert held.wait(10)
    manager.load_images(b)
    second = weakref.ref(manager.loader_thread)
    try:
        assert first() is not None and first().isRunning()  # kept while it runs
    finally:
        release.set()

    assert_freed(first, "the superseded loader thread", pump=True, timeout=WAIT_S)
    assert_freed(second, "the newer loader thread", pump=True, timeout=WAIT_S)
    assert sorted(manager.images) == sorted(B_STEMS)  # the superseded scan delivered nothing
    assert manager.loader_thread is None


def test_a_scan_that_ended_unread_is_kept_until_its_end_is_handled(app, tmp_path, no_gc):
    """A scan can end while its result and its end still wait in the event queue; opening
    another folder then must not free its thread under them (the reference cycle used to
    keep it): it goes when its queued end has run."""
    a, b = _project(tmp_path / "a", A_STEMS), _project(tmp_path / "b", B_STEMS)
    manager = im.ImageManager(None)
    manager.load_images(a)
    first = weakref.ref(manager.loader_thread)
    assert first().wait(10000)  # ended; nothing of it was handled yet (no event loop ran)

    manager.load_images(b)

    assert first() is not None
    assert_freed(first, "the ended scan's thread", pump=True, timeout=WAIT_S)
    assert _pump(lambda: manager.loader_thread is None)
    assert sorted(manager.images) == sorted(B_STEMS)  # a's late result was dropped


# ------------------------------------------------------------------ late results of an old scan
class _Window:
    """The main-window slots ImageManager calls, recorded."""

    def __init__(self) -> None:
        self.loaded: list[list[str]] = []
        self.failed: list[tuple] = []
        self.empty: list[str] = []

    def update_loading_progress(self, percent) -> None:
        pass

    def on_loading_finished(self, images, *_args) -> None:
        self.loaded.append(sorted(images))

    def on_loading_failed(self, folder, message) -> None:
        self.failed.append((folder, message))

    def on_loading_empty(self, folder) -> None:
        self.empty.append(folder)


def test_the_late_result_of_a_superseded_scan_is_dropped(app, tmp_path):
    """Folder A's scan has ended, its result still queued, when folder B is opened: only B is
    adopted, under B's folder (adopting A's late result put A's images under B's folder, and
    then B's under no folder: Generate and Measure would run on a mixed project)."""
    a, b = _project(tmp_path / "a", A_STEMS), _project(tmp_path / "b", B_STEMS)
    window = _Window()
    manager = im.ImageManager(window)
    manager.load_images(a)
    assert manager.loader_thread.wait(10000)  # ended; its result waits in the event queue

    manager.load_images(b)
    assert _pump(lambda: manager.loader_thread is None and window.loaded)

    assert window.loaded == [sorted(B_STEMS)] and window.failed == []
    assert sorted(manager.images) == sorted(B_STEMS)
    assert os.path.normcase(manager.original_folder) == os.path.normcase(b)


def test_the_late_failure_of_a_superseded_scan_is_ignored(app, tmp_path):
    """A scan of a folder that is gone has failed, its message still queued, when folder B is
    opened: the window hears nothing of the old failure, and B loads under its folder."""
    b = _project(tmp_path / "b", B_STEMS)
    window = _Window()
    manager = im.ImageManager(window)
    manager.load_images(str(tmp_path / "gone"))
    assert manager.loader_thread.wait(10000)  # failed; its message waits in the event queue

    manager.load_images(b)
    assert _pump(lambda: manager.loader_thread is None and window.loaded)

    assert window.failed == [] and window.loaded == [sorted(B_STEMS)]
    assert os.path.normcase(manager.original_folder) == os.path.normcase(b)
