"""The batch engine shared by mask and skeleton generation (``app/inference/batch_engine.py``).

Promises: batches keep the serial composition and order; an input that cannot be decoded is a
recorded failure and never an output, its slot zero-filled; cancel stops at a batch boundary
and keeps the finished outputs; progress counts outputs on disk, never goes backwards and ends
at the total, and a progress callback that raises changes nothing (R3: an input was listed as
written and failed); the caller's report is filled in place also when a step raises (R7); a
GPU run gives its pool threads one torch thread and restores the count (S12-01); failures are
logged a few at a time (C6-02). Out of memory and fixed batches: ``test_batch_engine_memory.py``;
failed saves: ``test_batch_engine_saves.py``. CPU stubs, no checkpoint, no GPU.
"""

from __future__ import annotations

import logging
import os
import time

import pytest
import torch
from PIL import Image

from app.inference import batch_engine
from app.inference.batch_engine import BatchReport, BatchSpec, log_failed_inputs, run_batches
from app.inference.batch_pipeline import FAILURE_LOG_LIMIT
from tests.generation_stubs_support import NAMES, _load, _make_images, _outputs, _spec, _step

ENGINE_LOGGER = "app.inference.batch_engine"


OPENMP_ONLY = pytest.mark.skipif(
    not torch.backends.openmp.is_available(), reason="the per-thread count is OpenMP's"
)


def test_unreadable_input_is_a_recorded_failure_not_an_output(tmp_path):
    clean = _make_images(str(tmp_path / "clean"), NAMES)
    ref = run_batches(clean, _spec(str(tmp_path / "ref")), batch_size=3)
    assert ref.failed == [] and len(ref.written) == 7

    dirty = _make_images(str(tmp_path / "dirty"), NAMES + ["bad.jpg"], corrupt={"bad.jpg"})
    report = run_batches(dirty, _spec(str(tmp_path / "out")), batch_size=3)

    assert [name for name, _ in report.failed] == ["bad.jpg"]
    assert report.failed_names == ["bad.jpg"]
    assert not os.path.exists(tmp_path / "out" / "bad.png")
    assert _outputs(tmp_path / "out") == _outputs(tmp_path / "ref")
    assert report.total == 8 and report.done == 8 and not report.cancelled


def test_batches_keep_serial_composition_and_order(tmp_path):
    paths = _make_images(str(tmp_path / "in"), NAMES)
    seen = []

    def recording_step(x):
        seen.append(x.shape[0])
        return _step(x)

    run_batches(paths, _spec(str(tmp_path / "out"), step=recording_step), batch_size=3)
    assert seen == [3, 3, 1]


def test_the_report_records_the_batch_shapes(tmp_path):
    paths = _make_images(str(tmp_path / "in"), NAMES)
    report = run_batches(paths, _spec(str(tmp_path / "out")), batch_size=3)
    assert report.batch_shapes == {3: 6, 1: 1}
    assert report.oom_events == []
    assert report.summary()["batch_shapes"] == {"3": 6, "1": 1}


def test_zero_filled_slots_count_in_the_forward_shape(tmp_path):
    dirty = _make_images(str(tmp_path / "dirty"), NAMES + ["bad.jpg"], corrupt={"bad.jpg"})
    report = run_batches(dirty, _spec(str(tmp_path / "out")), batch_size=3)
    assert report.batch_shapes == {3: 6, 2: 2}


def test_an_empty_report_has_empty_shapes():
    summary = BatchReport().summary()
    assert summary["batch_shapes"] == {}
    assert summary["oom_events"] == []


def test_cancel_stops_at_a_batch_boundary_and_keeps_finished_outputs(tmp_path):
    paths = _make_images(str(tmp_path / "in"), NAMES)
    calls = []

    def step(x):
        calls.append(x.shape[0])
        return _step(x)

    report = run_batches(
        paths,
        _spec(str(tmp_path / "out"), step=step),
        batch_size=2,
        should_stop=lambda: len(calls) >= 1,
    )
    assert report.cancelled
    assert calls == [2]
    assert len(report.written) == 2
    written = sorted(os.listdir(tmp_path / "out"))
    assert written == ["img00.png", "img01.png"]
    for name in written:  # complete, readable files only
        with Image.open(tmp_path / "out" / name) as im:
            im.load()


def test_progress_is_throttled_monotonic_and_complete(tmp_path):
    paths = _make_images(str(tmp_path / "in"), NAMES)
    ticks = []
    run_batches(
        paths,
        _spec(str(tmp_path / "out")),
        batch_size=3,
        progress=lambda done, total: ticks.append((done, total)),
    )
    dones = [d for d, _ in ticks]
    assert dones == sorted(dones) and dones[-1] == 7
    assert all(total == 7 for _, total in ticks)


def test_empty_input_is_an_empty_report(tmp_path):
    report = run_batches([], _spec(str(tmp_path / "out")), batch_size=4)
    assert report == BatchReport(total=0, batch_size=4, effective_batch_size=4)


class _BrokenProgress:
    """A progress callback that raises on its first ``fail_first`` calls."""

    def __init__(self, fail_first=1):
        self.fail_first = fail_first
        self.calls = []

    def __call__(self, *args):
        self.calls.append(args)
        if len(self.calls) <= self.fail_first:
            raise RuntimeError("progress listener is gone (simulated)")


def test_a_raising_progress_callback_leaves_a_written_input_written(tmp_path, caplog):
    paths = _make_images(str(tmp_path / "in"), NAMES[:3])
    progress = _BrokenProgress()

    with caplog.at_level(logging.WARNING):
        report = batch_engine.run_batches(
            paths, _spec(str(tmp_path / "out")), batch_size=3, progress=progress
        )

    assert report.failed == []  # never both written and failed
    assert report.written == sorted(NAMES[:3])
    assert report.done == report.total == 3
    assert len(progress.calls) == 3  # later updates still reach the callback
    assert "progress listener is gone" in caplog.text  # logged, not swallowed silently


def test_a_raising_progress_callback_never_ends_the_run_on_an_unreadable_input(tmp_path):
    names = [*NAMES[:3], "bad.jpg"]
    paths = _make_images(str(tmp_path / "in"), names, corrupt={"bad.jpg"})
    progress = _BrokenProgress(fail_first=10)  # every call raises

    report = batch_engine.run_batches(
        paths, _spec(str(tmp_path / "out")), batch_size=2, progress=progress
    )

    assert report.failed_names == ["bad.jpg"]
    assert report.written == sorted(NAMES[:3])
    assert report.status == "completed" and report.done == report.total == 4


def test_progress_updates_arrive_in_order_across_writer_threads(tmp_path):
    # Saves finish on several writer threads. The count was taken under a lock but the
    # callback ran after releasing it, so a slow delivery of an early count could land
    # after later ones: the progress bar went backwards and could end on "1 of 7".
    paths = _make_images(str(tmp_path / "in"), NAMES)
    applied = []

    def slow_first_update(done, total):
        if done == 1:
            time.sleep(0.2)  # e.g. the GUI thread busy when the first signal is emitted
        applied.append(done)

    batch_engine.run_batches(
        paths,
        _spec(str(tmp_path / "out")),
        batch_size=len(NAMES),
        io_threads=4,
        progress=slow_first_update,
    )

    assert applied == sorted(applied) and applied[-1] == len(NAMES)


@pytest.mark.parametrize("fail_first", [1, 3])
def test_progress_still_ends_at_the_total_after_a_failing_update(tmp_path, fail_first):
    paths = _make_images(str(tmp_path / "in"), NAMES)
    progress = _BrokenProgress(fail_first=fail_first)

    batch_engine.run_batches(paths, _spec(str(tmp_path / "out")), batch_size=3, progress=progress)

    dones = [done for done, _total in progress.calls]
    assert dones == sorted(dones) and dones[-1] == len(NAMES)


def test_the_summary_states_the_outcome():
    assert BatchReport().summary()["status"] == "completed"
    assert BatchReport(cancelled=True).summary()["status"] == "cancelled"
    summary = BatchReport(cancelled=True, error="RuntimeError: boom").summary()
    assert summary["status"] == "failed" and summary["error"] == "RuntimeError: boom"
    assert BatchReport().summary()["error"] is None


def test_run_batches_fills_the_given_report_even_when_a_step_raises(tmp_path):
    paths = _make_images(str(tmp_path / "in"), NAMES)
    out = str(tmp_path / "out")
    calls = []

    def step(x):
        calls.append(len(x))
        if len(calls) == 2:
            raise RuntimeError("boom")
        return _step(x)

    report = BatchReport()
    with pytest.raises(RuntimeError, match="boom"):
        run_batches(paths, _spec(out, step=step), batch_size=3, report=report)

    assert report.written == sorted(os.listdir(out)) == NAMES[:3]  # what is on disk
    assert report.total == 7 and report.done == 3
    assert report.batch_size == report.effective_batch_size == 3
    assert report.elapsed_s > 0


def test_a_run_cancelled_as_a_step_raises_still_records_the_cancel(tmp_path):
    # Cancel pressed during a batch that then fails: the error decides the status
    # ("failed"), and the cancel flag still tells the truth.
    paths = _make_images(str(tmp_path / "in"), NAMES)
    out = str(tmp_path / "out")
    cancelled, calls = [], []

    def step(x):
        calls.append(len(x))
        if len(calls) == 2:
            cancelled.append(True)  # Cancel pressed during the second batch...
            raise RuntimeError("boom")  # ...which then fails
        return _step(x)

    report = BatchReport()
    with pytest.raises(RuntimeError, match="boom"):
        run_batches(
            paths,
            _spec(out, step=step),
            batch_size=3,
            report=report,
            should_stop=lambda: bool(cancelled),
        )
    assert report.done < report.total
    assert report.cancelled is True


def test_run_batches_returns_the_report_it_was_given(tmp_path):
    paths = _make_images(str(tmp_path / "in"), NAMES)
    report = BatchReport()

    returned = run_batches(paths, _spec(str(tmp_path / "out")), batch_size=3, report=report)

    assert returned is report
    assert len(report.written) == 7 and report.status == "completed"


def _thread_recording_spec(seen):
    def load(path):
        seen.add(("load", torch.get_num_threads()))
        return _load(path)

    def save(arr, path):
        seen.add(("save", torch.get_num_threads()))

    return BatchSpec(load=load, step=_step, save=save)


@OPENMP_ONLY
def test_torch_threads_hold_in_the_pool_threads_and_are_restored(tmp_path):
    paths = _make_images(str(tmp_path / "in"), NAMES)
    before = torch.get_num_threads()
    seen: set[tuple[str, int]] = set()
    report = run_batches(paths, _thread_recording_spec(seen), batch_size=3, torch_threads=1)
    assert len(report.written) == 7
    assert seen == {("load", 1), ("save", 1)}
    assert torch.get_num_threads() == before


def test_without_torch_threads_the_count_is_left_alone(tmp_path):
    paths = _make_images(str(tmp_path / "in"), NAMES)
    before = torch.get_num_threads()
    seen: set[tuple[str, int]] = set()
    run_batches(paths, _thread_recording_spec(seen), batch_size=3)
    assert seen == {("load", before), ("save", before)}
    assert torch.get_num_threads() == before


@OPENMP_ONLY
def test_torch_threads_are_restored_when_a_step_raises(tmp_path):
    paths = _make_images(str(tmp_path / "in"), NAMES)
    before = torch.get_num_threads()

    def broken_step(x):
        raise ValueError("model broke")

    with pytest.raises(ValueError, match="model broke"):
        run_batches(
            paths, _spec(str(tmp_path / "out"), step=broken_step), batch_size=3, torch_threads=1
        )
    assert torch.get_num_threads() == before


def _messages(caplog, text):
    return [r for r in caplog.records if text in r.getMessage()]


def test_many_unreadable_inputs_log_the_first_few_then_a_count(tmp_path, caplog):
    names = [f"bad{i:02d}.jpg" for i in range(FAILURE_LOG_LIMIT + 5)]
    paths = _make_images(str(tmp_path / "in"), names, corrupt=set(names))
    with caplog.at_level(logging.WARNING, logger=ENGINE_LOGGER):
        report = run_batches(paths, _spec(str(tmp_path / "out")), batch_size=4)
    assert len(report.failed) == len(names)  # every failure is still in the report
    unreadable = _messages(caplog, "Unreadable input")
    assert len(unreadable) == FAILURE_LOG_LIMIT
    # level and source line as before: the engine's own line, not the limiter's
    assert all(r.levelno == logging.WARNING for r in unreadable)
    assert all(r.pathname.endswith("batch_engine.py") for r in unreadable)
    [more] = _messages(caplog, "more unreadable inputs")
    assert more.getMessage().startswith("5 more")


def test_a_few_unreadable_inputs_are_all_logged_without_a_count(tmp_path, caplog):
    paths = _make_images(str(tmp_path / "in"), NAMES + ["bad.jpg"], corrupt={"bad.jpg"})
    with caplog.at_level(logging.WARNING, logger=ENGINE_LOGGER):
        run_batches(paths, _spec(str(tmp_path / "out")), batch_size=3)
    assert len(_messages(caplog, "Unreadable input")) == 1
    assert _messages(caplog, " more ") == []


def test_the_end_of_run_line_names_the_first_failures_and_the_record(caplog):
    log = logging.getLogger("tests.end_of_run")
    report = BatchReport(failed=[(f"img{i}.jpg", "OSError: broken") for i in range(8)])
    with caplog.at_level(logging.WARNING, logger=log.name):
        log_failed_inputs(log, "Mask generation", report, "out/_sprouts_run.json")
        log_failed_inputs(log, "Mask generation", report, None)
        log_failed_inputs(log, "Mask generation", BatchReport(), "out/_sprouts_run.json")
    first, second = (r.getMessage() for r in caplog.records)  # nothing for no failure
    assert first.startswith("Mask generation: 8 input(s) failed: img0.jpg: OSError: broken; ")
    assert "img4.jpg" in first and "img5.jpg" not in first and "and 3 more" in first
    assert first.endswith("; every one is listed in out/_sprouts_run.json")
    assert "listed in" not in second  # no record was written: it is not named
    assert all(r.pathname.endswith("test_batch_engine.py") for r in caplog.records)
