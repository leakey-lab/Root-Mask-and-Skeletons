"""The app-wide batch-job registry (``app/handlers/job_registry.py``).

One batch job at a time: ``try_start`` refuses a second one, so Measure cannot read a
half-written ``skeletons/`` folder while generation runs. The registry clears itself on
Qt's own ``finished`` (also when a worker shadows ``finished``), a stale ``finished`` never
clears a later job, ``stop_all`` cancels and waits, and ``ended_name`` tells the job-end
slot which job ended; once ``shutdown`` (the window is closing) it refuses every job.

Offscreen Qt. Threads are real ``QThread`` s held on a ``threading.Event``, so the tests
never race them.
"""

from __future__ import annotations

import threading
import time

import pytest
from PyQt6.QtCore import QCoreApplication, QEventLoop, QThread, pyqtSignal
from PyQt6.QtWidgets import QApplication

from app.handlers.job_registry import JobRegistry
from app.workers import GuardedThread


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


class _StubbornThread(QThread):
    """Plain QThread (custom finished) that ignores interruption until released."""

    finished = pyqtSignal(dict)

    def __init__(self) -> None:
        super().__init__()
        self.gate = threading.Event()

    def run(self) -> None:
        self.gate.wait(10)


class _PoliteThread(QThread):
    """Plain QThread that honours requestInterruption()."""

    def run(self) -> None:
        while not self.isInterruptionRequested():
            time.sleep(0.002)


class Job(QThread):
    """A batch job that ends at once."""

    def run(self) -> None:
        pass


def until(condition, timeout_s: float = 5.0) -> bool:
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        if condition():
            return True
        QApplication.processEvents()
    return condition()


def run_job(jobs: JobRegistry, name: str) -> None:
    """Start ``name`` through the registry and wait until it has ended (busy False)."""
    ended = []
    jobs.busy_changed.connect(lambda busy: ended.append(busy) if not busy else None)
    thread = Job()
    assert jobs.try_start(name, thread)
    assert until(lambda: ended)
    thread.wait()


@pytest.fixture
def registry(qapp):
    reg = JobRegistry()
    states: list[bool] = []
    reg.busy_changed.connect(states.append)
    reg.states = states  # test-only attribute for assertions
    return reg


def test_try_start_runs_one_job_and_rejects_a_second(registry):
    first, second = _BlockingThread(), _BlockingThread()
    assert registry.try_start("Generating masks", first) is True
    assert first.started_work.wait(5)
    assert registry.is_busy() is True
    assert registry.running_name() == "Generating masks"
    assert registry.try_start("Measuring", second) is False
    assert second.isRunning() is False
    first.gate.set()
    assert first.wait(5000)
    _pump_until(lambda: not registry.is_busy())
    assert registry.running_name() is None
    assert registry.states == [True, False]


def test_registry_clears_when_subclass_shadows_finished(registry):
    thread = _StubbornThread()  # custom finished(dict) that is never emitted
    assert registry.try_start("Loading images", thread)
    thread.gate.set()
    assert thread.wait(5000)
    _pump_until(lambda: not registry.is_busy())
    assert registry.try_start("Next job", _ResultThread()) is True
    _pump_until(lambda: not registry.is_busy())


def test_cancel_all_requests_interruption(registry):
    guarded, plain = _BlockingThread(), _PoliteThread()
    rec = _Recorder(guarded)
    assert registry.try_start("Generating skeletons", guarded)
    assert guarded.started_work.wait(5)
    registry.cancel_all()
    assert guarded.wait(5000)
    _pump_until(lambda: rec.cancelled and not registry.is_busy())

    assert registry.try_start("Measuring", plain)
    registry.cancel_all()
    assert plain.wait(5000)
    _pump_until(lambda: not registry.is_busy())


def test_stop_all_waits_and_clears_synchronously(registry):
    thread = _BlockingThread()
    assert registry.try_start("Generating masks", thread)
    assert thread.started_work.wait(5)
    assert registry.stop_all(timeout_ms=5000) is True
    assert thread.isFinished()
    assert registry.is_busy() is False
    assert registry.states == [True, False]
    # The queued finished of the stopped job must not clear a job started after it.
    follow_up = _BlockingThread()
    assert registry.try_start("Measuring", follow_up)
    _flush()
    assert registry.running_name() == "Measuring"
    follow_up.gate.set()
    assert follow_up.wait(5000)
    _pump_until(lambda: not registry.is_busy())


def test_stop_all_reports_a_job_that_does_not_stop(registry):
    thread = _StubbornThread()
    assert registry.try_start("Stubborn", thread)
    try:
        assert registry.stop_all(timeout_ms=50) is False
        assert registry.running_name() == "Stubborn"
    finally:
        thread.gate.set()
        assert thread.wait(5000)
    _pump_until(lambda: not registry.is_busy())


def test_stop_all_without_a_job_is_true(registry):
    assert registry.stop_all() is True
    assert registry.states == []


def test_the_job_end_slot_can_tell_which_job_ended(app):
    jobs = JobRegistry()
    seen = []
    jobs.busy_changed.connect(
        lambda busy: seen.append((busy, jobs.running_name(), jobs.ended_name()))
    )
    assert jobs.ended_name() is None

    run_job(jobs, "Generating masks")

    assert seen == [(True, "Generating masks", None), (False, None, "Generating masks")]


class _LastBatchCancelled(GuardedThread):
    """Cancel pressed during the final batch: the engine reports completion anyway."""

    def __init__(self):
        super().__init__()
        self.completed = False

    def work(self):
        self.cancel()
        self.completed = True  # done == total, so the run "completed"


def test_job_registry_refuses_new_jobs_once_shut_down(qapp):
    registry = JobRegistry()
    registry.shutdown()
    thread = _LastBatchCancelled()
    assert registry.is_closing()
    assert registry.try_start("Generating skeletons", thread) is False
    assert not thread.isRunning()
