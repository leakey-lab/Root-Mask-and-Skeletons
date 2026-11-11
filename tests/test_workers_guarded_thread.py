"""The worker error boundary of ``app/workers.py``: ``GuardedThread`` and ``describe_error``.

Exactly one outcome per run: the subclass's own result signal, ``cancelled`` (after
``cancel()`` / ``JobCancelled``), or ``failed(message)`` for anything else that ends
``work()``, ``SystemExit`` and ``KeyboardInterrupt`` included (unreported, the job registry
would hold the job for good). Nothing escapes ``run()``, not even a broken ``describe()``
or ``__str__``. A user condition (a file open in another program, an output folder that
takes no files) keeps its own message and is logged without a traceback.

Offscreen Qt. Threads are real ``QThread`` s; tests that act while a job runs hold it on a
``threading.Event`` so they never race it. Queued signals are flushed with ``_pump_until``.
"""

from __future__ import annotations

import logging
import sys
import threading
import time

import pytest
from PyQt6.QtCore import QCoreApplication, QEventLoop, pyqtSignal

from app.workers import GuardedThread, JobCancelled, describe_error


def _pump_until(predicate, timeout_s: float = 5.0) -> None:
    """Process queued events until ``predicate()`` is true (or fail after the timeout)."""
    deadline = time.monotonic() + timeout_s
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached before timeout")
        QCoreApplication.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 20)
        time.sleep(0.002)


def _flush() -> None:
    for _ in range(5):
        QCoreApplication.processEvents()


class _Recorder:
    """Collects a GuardedThread's terminal signals."""

    def __init__(self, thread: GuardedThread) -> None:
        self.failed: list[str] = []
        self.cancelled = 0
        thread.failed.connect(self.failed.append)
        thread.cancelled.connect(self._on_cancelled)

    def _on_cancelled(self) -> None:
        self.cancelled += 1


class _RaisingThread(GuardedThread):
    def __init__(self, exc: BaseException) -> None:
        super().__init__()
        self._exc = exc

    def work(self) -> None:
        raise self._exc


class _ResultThread(GuardedThread):
    """Shadows QThread.finished with a custom signature, like the app's workers."""

    finished = pyqtSignal(str)

    def work(self) -> None:
        self.finished.emit("ok")


class _BlockingThread(GuardedThread):
    """Waits on ``gate``; stops early (JobCancelled) when cancelled."""

    def __init__(self) -> None:
        super().__init__()
        self.gate = threading.Event()
        self.started_work = threading.Event()

    def work(self) -> None:
        self.started_work.set()
        while not self.gate.wait(0.005):
            self.check_cancelled()


# --- describe_error ----------------------------------------------------------------


def test_describe_error_formats_type_and_message():
    assert describe_error(ValueError("bad value")) == "ValueError: bad value"
    assert describe_error(MemoryError()) == "MemoryError"


def test_describe_error_survives_a_broken_str():
    class Unprintable(Exception):
        def __str__(self):
            raise RuntimeError("cannot render")

    assert describe_error(Unprintable()) == "Unprintable"


def test_file_locked_error_is_logged_without_a_traceback(app, caplog, tmp_path):
    from app.io_utils import FileLockedError
    from app.workers import GuardedThread, describe_error, is_user_condition

    exc = FileLockedError(tmp_path / "root_lengths.csv")

    class Locked(GuardedThread):
        def work(self):
            raise exc

    thread = Locked()
    messages = []
    thread.failed.connect(messages.append)
    with caplog.at_level(logging.WARNING, logger="app.handlers.worker_base"):
        thread.run()  # the error boundary, called synchronously
    assert messages == [str(exc)] == [describe_error(exc)]
    assert is_user_condition(exc) and not is_user_condition(ValueError("x"))
    records = [r for r in caplog.records if r.name == "app.handlers.worker_base"]
    assert [r.levelno for r in records] == [logging.WARNING]
    assert records[0].exc_info is None


def test_an_unwritable_output_folder_is_a_user_condition(tmp_path):
    """Generate's stop for a folder that takes no new files (C6-01) shows its own message,
    without the exception type and without a traceback in the log."""
    from app.io_utils import FolderNotWritableError
    from app.workers import describe_error, is_user_condition

    exc = FolderNotWritableError(tmp_path / "mask", "access denied", 13)
    assert is_user_condition(exc)
    assert describe_error(exc) == exc.message
    assert describe_error(exc).startswith(f"Cannot save files in {tmp_path / 'mask'}")


def test_a_broken_batch_engine_import_does_not_hide_a_locked_file(monkeypatch, tmp_path):
    """The user conditions are imported one at a time: one that cannot be imported must not
    turn a locked CSV into "FileLockedError: ..." with a traceback."""
    from app.io_utils import FileLockedError
    from app.workers import is_user_condition

    monkeypatch.setitem(sys.modules, "app.inference.batch_engine", None)  # its import fails
    exc = FileLockedError(tmp_path / "root_lengths.csv")
    assert is_user_condition(exc)
    assert describe_error(exc) == str(exc)


# --- GuardedThread -----------------------------------------------------------------


def test_exception_in_work_emits_failed_and_logs_traceback(qapp, caplog):
    thread = _RaisingThread(PermissionError(13, "Permission denied", "root_lengths.csv"))
    rec = _Recorder(thread)
    with caplog.at_level(logging.ERROR, logger="app.handlers.worker_base"):
        thread.start()
        assert thread.wait(5000)
        _pump_until(lambda: rec.failed)
    assert rec.failed == ["PermissionError: [Errno 13] Permission denied: 'root_lengths.csv'"]
    assert rec.cancelled == 0
    records = [r for r in caplog.records if r.exc_info]
    assert records and records[0].exc_info[0] is PermissionError
    assert "_RaisingThread" in records[0].getMessage()


def test_normal_completion_emits_only_the_subclass_signal(qapp):
    thread = _ResultThread()
    rec = _Recorder(thread)
    results: list[str] = []
    thread.finished.connect(results.append)
    thread.start()
    assert thread.wait(5000)
    _pump_until(lambda: results)
    _flush()
    assert results == ["ok"]
    assert (rec.failed, rec.cancelled) == ([], 0)


def test_cancel_stops_work_and_emits_cancelled(qapp):
    thread = _BlockingThread()
    rec = _Recorder(thread)
    thread.start()
    assert thread.started_work.wait(5)
    assert thread.is_cancelled() is False
    thread.cancel()
    assert thread.is_cancelled() is True
    assert thread.wait(5000)
    _pump_until(lambda: rec.cancelled)
    _flush()
    assert (rec.failed, rec.cancelled) == ([], 1)


def test_raising_job_cancelled_directly_counts_as_cancel(qapp):
    thread = _RaisingThread(JobCancelled())
    rec = _Recorder(thread)
    thread.start()
    assert thread.wait(5000)
    _pump_until(lambda: rec.cancelled)
    assert rec.failed == []


class _Abort(BaseException):
    """A BaseException that is not an Exception, SystemExit or KeyboardInterrupt."""


@pytest.mark.parametrize(
    ("exc", "message"),
    [
        (_Abort("stopped"), "_Abort: stopped"),
        (KeyboardInterrupt(), "KeyboardInterrupt"),
        (SystemExit(3), "SystemExit: 3"),
    ],
)
def test_a_base_exception_in_work_emits_failed(qapp, caplog, exc, message):
    # Unreported, the job registry would hold the job for good and refuse every later one.
    thread = _RaisingThread(exc)
    rec = _Recorder(thread)
    with caplog.at_level(logging.ERROR, logger="app.handlers.worker_base"):
        thread.start()
        assert thread.wait(5000)
        _pump_until(lambda: rec.failed)
    assert rec.failed == [message]
    assert rec.cancelled == 0
    assert any(r.exc_info and r.exc_info[0] is type(exc) for r in caplog.records)


def test_work_must_be_implemented(qapp):
    thread = GuardedThread()
    rec = _Recorder(thread)
    thread.start()
    assert thread.wait(5000)
    _pump_until(lambda: rec.failed)
    assert rec.failed[0].startswith("NotImplementedError")


class _WordedThread(_RaisingThread):
    """Words its own failure message, like the dashboard build worker."""

    def describe(self, exc: BaseException) -> str:
        return f"Could not finish: {exc}"


class _BrokenWordingThread(_RaisingThread):
    def describe(self, exc: BaseException) -> str:
        raise SystemExit("describe() itself failed")


@pytest.mark.parametrize("exc", [OSError("disk gone"), SystemExit("disk gone")])
def test_a_subclass_words_its_own_failure_message(qapp, caplog, exc):
    thread = _WordedThread(exc)
    rec = _Recorder(thread)
    with caplog.at_level(logging.ERROR, logger="app.handlers.worker_base"):
        thread.start()
        assert thread.wait(5000)
        _pump_until(lambda: rec.failed)
    assert rec.failed == ["Could not finish: disk gone"]
    assert rec.cancelled == 0
    assert any(r.exc_info and r.exc_info[0] is type(exc) for r in caplog.records)


def test_a_broken_describe_falls_back_to_describe_error(qapp):
    thread = _BrokenWordingThread(RuntimeError("boom"))
    rec = _Recorder(thread)
    thread.run()  # on this thread: whatever escapes run() would escape the worker thread
    _pump_until(lambda: rec.failed)
    assert rec.failed == ["RuntimeError: boom"]
