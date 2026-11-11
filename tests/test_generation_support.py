"""tests/generation_support.py: waiting for a worker thread without fixed sleeps; and
conftest's ``cpu_runtime``, which must never hand a test a runtime cached before it."""

import time

import pytest
import torch
from PyQt6.QtCore import QObject, Qt, QThread, pyqtSlot

from app.inference import runtime
from tests.generation_support import (
    wait_for_thread,
)


class _Receiver(QObject):
    """Lives on the GUI thread: a signal of a worker reaches it queued."""

    def __init__(self):
        super().__init__()
        self.seen: list[str] = []

    @pyqtSlot()
    def on_finished(self):
        self.seen.append("finished")


class _Sleeper(QThread):
    def __init__(self, seconds: float):
        super().__init__()
        self.seconds = seconds

    def run(self):
        time.sleep(self.seconds)


@pytest.fixture
def earlier_runtime():
    """A runtime left in the cache before ``cpu_runtime`` is set up, as an earlier test may
    leave one; after ``cpu_runtime``'s teardown none is cached."""
    # Its own patch, undone at once: the test's CPU device must come from cpu_runtime's.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(torch.cuda, "is_available", lambda: False)  # no CUDA set-up here
        runtime.get_runtime.cache_clear()
        cached = runtime.get_runtime()
    yield cached
    assert runtime.get_runtime.cache_info().currsize == 0


def test_a_slot_queued_by_a_slow_finished_signal_is_delivered(qapp):
    """``isFinished()`` holds while ``finished`` is still being emitted: here a direct slot
    keeps the emission busy before the queued one is posted. ``wait_for_thread`` waits for
    the emission to end, then delivers."""
    thread, receiver = _Sleeper(0.0), _Receiver()
    thread.finished.connect(lambda: time.sleep(0.3), type=Qt.ConnectionType.DirectConnection)
    thread.finished.connect(receiver.on_finished)
    thread.start()

    wait_for_thread(thread, qapp)

    assert receiver.seen == ["finished"]


@pytest.mark.wallclock
def test_wait_for_thread_fails_when_the_thread_outlives_the_timeout(qapp):
    thread = _Sleeper(1.0)
    thread.start()
    started = time.monotonic()
    try:
        with pytest.raises(AssertionError):
            wait_for_thread(thread, qapp, timeout=0.1)
        assert time.monotonic() - started < 0.9
    finally:
        thread.wait()


def test_cpu_runtime_resolves_a_new_runtime_on_the_cpu(earlier_runtime, cpu_runtime):
    assert cpu_runtime.device.type == "cpu" and not cpu_runtime.is_cuda
    assert cpu_runtime is not earlier_runtime  # the cache was cleared first
    assert runtime.get_runtime() is cpu_runtime  # the runtime the code under test gets
