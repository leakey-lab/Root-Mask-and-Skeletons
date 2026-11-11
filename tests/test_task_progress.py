"""The task progress row (``app/gui/task_progress.py``) and its remaining-time estimate.

Promises: ``<operation> — <progress> · ETA``; item counts with thousands separators and the
matching percent; Cancel shows only for a cancellable task and runs its callback once; after
Cancel the row says ``<operation> — cancelling…`` until the job ends, also when the batch in
flight still reports or the job starts its next stage; a pending hide never hides the next
task; the state signal says working / error / idle. The ETA waits for enough time and
progress, restarts when progress goes back, and is formatted in minutes and hours.
"""

from __future__ import annotations

import pytest

from app.gui.task_progress import EtaEstimator, TaskProgressWidget, format_eta
from tests.qt_teardown import flush_deleted


class _Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def test_eta_from_a_steady_rate():
    clock = _Clock()
    eta = EtaEstimator(clock=clock)
    eta.reset()
    clock.t = 10.0
    eta.update(0.1)
    clock.t = 20.0
    assert eta.update(0.2) == pytest.approx(80.0)


def test_eta_is_none_until_enough_time_and_progress():
    clock = _Clock()
    eta = EtaEstimator(clock=clock)
    eta.reset()
    clock.t = 1.0
    assert eta.update(0.5) is None  # under 2 s
    clock = _Clock()
    eta = EtaEstimator(clock=clock)
    eta.reset()
    clock.t = 30.0
    assert eta.update(0.01) is None  # under 2 %
    clock.t = 40.0
    assert eta.update(1.0) is None  # done


def test_eta_restarts_when_progress_goes_backwards():
    clock = _Clock()
    eta = EtaEstimator(clock=clock)
    eta.reset()
    clock.t = 10.0
    eta.update(0.5)
    clock.t = 11.0
    assert eta.update(0.1) is None


@pytest.mark.parametrize(
    ("seconds", "text"),
    [
        (20, "ETA <1 min"),
        (90, "ETA 2 min"),
        (3 * 60, "ETA 3 min"),
        (2 * 3600 + 300, "ETA 2 h 5 min"),
    ],
)
def test_format_eta(seconds, text):
    assert format_eta(seconds) == text


def test_public_api_is_kept(app):
    tp = TaskProgressWidget()
    tp.start("Generating masks")
    assert not tp.isHidden()
    tp.set_progress(37)
    assert tp.label.text() == "Generating masks — 37%"
    assert tp.bar.value() == 37
    tp.set_operation("Extracting skeletons")
    assert tp.label.text().startswith("Extracting skeletons — 37%")
    tp.fail("Boom")
    assert tp.label.text() == "Boom"
    tp.start("x")
    tp.finish()
    assert tp.isHidden()


def test_set_counts_shows_done_of_total_and_eta(app):
    clock = _Clock()
    tp = TaskProgressWidget(clock=clock)
    tp.start("Generating masks")
    clock.t = 10.0
    tp.set_counts(4_617, 46_178)
    clock.t = 20.0
    tp.set_counts(9_234, 46_178)
    text = tp.label.text()
    assert text.startswith("Generating masks — 9,234 / 46,178")
    assert "· ETA " in text
    assert tp.bar.value() == 19  # progress_percent floors


def test_cancel_calls_the_callback_once(app):
    calls, signals = [], []
    tp = TaskProgressWidget()
    tp.canceled.connect(lambda: signals.append(1))
    tp.start("Generating masks", on_cancel=lambda: calls.append(1))
    assert not tp.cancel_button.isHidden()
    tp.cancel_button.click()
    assert calls == [1] and signals == [1]
    assert not tp.cancel_button.isEnabled()
    assert tp.label.text() == "Generating masks — cancelling…"


def test_cancel_hidden_when_not_cancelable(app):
    tp = TaskProgressWidget()
    tp.start("Loading images")
    assert tp.cancel_button.isHidden()


def test_single_row_fits_the_statusline(app):
    tp = TaskProgressWidget()
    tp.start("Generating masks", cancelable=True)
    assert tp.sizeHint().height() <= 24


def test_state_changes_are_signalled(app):
    tp = TaskProgressWidget()
    states = []
    tp.state_changed.connect(states.append)
    tp.start("x")
    tp.fail("y")
    tp.start("z")
    tp.finish()
    assert states == ["working", "error", "working", "idle"]


_HIDE_MS = 40  # the patched hide delay: fast, and the control below proves it is used


def _pump(ms: int) -> None:
    from PyQt6.QtCore import QEventLoop, QTimer

    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec()


def _ended_task(end: str) -> TaskProgressWidget:
    tp = TaskProgressWidget()
    tp.start("first")
    if end == "finish":
        tp.finish("First done")  # schedules a hide after FINISH_HIDE_MS
    else:
        tp.fail("First failed")  # schedules a hide after FAIL_HIDE_MS
    return tp


@pytest.mark.parametrize(("end", "delay"), [("finish", "FINISH_HIDE_MS"), ("fail", "FAIL_HIDE_MS")])
def test_pending_hide_never_hides_the_next_task(app, monkeypatch, end, delay):
    # RB-16: the hide is one member timer that start() stops, never an untracked
    # singleShot that would hide the next task's row.
    monkeypatch.setattr(f"app.gui.task_progress.{delay}", _HIDE_MS)
    control = _ended_task(end)  # no next task: the row hides after the patched delay,
    _pump(_HIDE_MS + 300)  # so the same pump below outlasts the pending hide
    assert control.isHidden()
    control.deleteLater()

    tp = _ended_task(end)
    tp.start("second")
    _pump(_HIDE_MS + 300)
    assert not tp.isHidden()
    assert tp.label.text().startswith("second")
    tp.deleteLater()


class _Worker:
    def __init__(self) -> None:
        self.cancels = 0

    def cancel(self) -> None:
        self.cancels += 1


@pytest.fixture
def progress(app):
    widget = TaskProgressWidget()
    yield widget
    widget.deleteLater()
    del widget
    flush_deleted()


def test_progress_after_cancel_keeps_the_cancelling_text(progress):
    worker = _Worker()
    progress.start("Generating masks", cancelable=True, on_cancel=worker.cancel)
    progress.set_counts(1_234, 46_178)
    progress.cancel_button.click()
    assert progress.label.text() == "Generating masks — cancelling…"

    progress.set_counts(1_235, 46_178)  # the batch in flight still reports
    progress.set_progress(3)
    assert progress.label.text() == "Generating masks — cancelling…"
    assert progress.bar.value() == 3  # the bar still follows the last batch
    assert not progress.cancel_button.isEnabled()
    assert worker.cancels == 1


def test_a_new_stage_of_the_cancelled_job_stays_cancelling(progress):
    worker = _Worker()
    progress.start("Loading the mask model", cancelable=True, on_cancel=worker.cancel)
    progress.cancel_button.click()
    # The worker reports its next stage before it sees the cancel (same job, same cancel).
    progress.start("Generating masks", cancelable=True, on_cancel=worker.cancel)
    assert progress.label.text() == "Generating masks — cancelling…"
    assert not progress.cancel_button.isEnabled()
    assert not progress.cancel_button.isHidden()
    assert worker.cancels == 1


def test_the_end_of_a_cancelled_job_is_shown_and_the_next_job_starts_clean(progress):
    worker = _Worker()
    progress.start("Generating skeletons", cancelable=True, on_cancel=worker.cancel)
    progress.cancel_button.click()
    progress.finish("Skeleton generation cancelled after 12 of 40 images")
    assert progress.label.text() == "Skeleton generation cancelled after 12 of 40 images"

    other = _Worker()
    progress.start("Measuring root length", cancelable=True, on_cancel=other.cancel)
    assert progress.cancel_button.isEnabled()
    progress.set_progress(5)
    assert progress.label.text() == "Measuring root length — 5%"


def test_another_job_after_a_cancel_is_not_shown_as_cancelling(progress):
    first, second = _Worker(), _Worker()
    progress.start("Generating masks", cancelable=True, on_cancel=first.cancel)
    progress.cancel_button.click()
    progress.fail("Mask generation failed")
    progress.start("Measuring root area", cancelable=True, on_cancel=second.cancel)
    assert progress.cancel_button.isEnabled()
    progress.set_counts(1, 10)
    assert progress.label.text().startswith("Measuring root area — 1 / 10")


def test_cancelling_without_an_operation_name(progress):
    progress.start("", cancelable=True, on_cancel=lambda: None)
    progress.cancel_button.click()

    assert progress.label.text() == "Cancelling…"
