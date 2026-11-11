"""Out of GPU memory and fixed batches in the batch engine (``app/inference/batch_engine.py``).

``auto`` (PI-07, R5): CUDA out of memory halves the batch and retries down to one image; the
size that worked is kept; a slice at least as long as one that already ran out of memory is
split untried, so the slices that run are the plain halving's (``_legacy_halving``, the
reference) and outputs stay bit-identical; an image out of memory alone is one recorded
failure with the original reason. ``fixed`` (O4): every forward has exactly ``batch_size``
rows (the tail zero-padded, padding never saved); out of memory stops the run, no retry.
CPU only: out of memory is simulated.
"""

from __future__ import annotations

import logging
import os

import pytest
import torch

from app.inference import batch_engine
from app.inference.batch_engine import (
    FIXED_OOM_MESSAGE,
    BatchReport,
    FixedBatchOutOfMemory,
    OomBackoff,
    run_batches,
)
from app.inference.batch_pipeline import FAILURE_LOG_LIMIT
from tests.generation_stubs_support import NAMES, _make_images, _outputs, _spec, _step

ENGINE_LOGGER = "app.inference.batch_engine"


def _messages(caplog, text):
    return [r for r in caplog.records if text in r.getMessage()]


def test_out_of_memory_halves_the_batch_and_writes_everything(tmp_path):
    paths = _make_images(str(tmp_path / "in"), NAMES)
    sizes = []

    def oom_step(x):
        sizes.append(x.shape[0])
        if x.shape[0] > 2:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory (stub)")
        return _step(x)

    report = run_batches(paths, _spec(str(tmp_path / "out"), step=oom_step), batch_size=4)
    assert report.failed == []
    assert len(report.written) == 7
    assert report.effective_batch_size == 2
    run_batches(paths, _spec(str(tmp_path / "ref")), batch_size=4)
    assert _outputs(tmp_path / "out") == _outputs(tmp_path / "ref")
    # the working size is remembered: later batches do not retry at 4
    assert sizes.count(4) == 1


def test_the_report_records_out_of_memory_events(tmp_path):
    paths = _make_images(str(tmp_path / "in"), NAMES)

    def oom_step(x):
        if x.shape[0] > 2:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory (stub)")
        return _step(x)

    report = run_batches(paths, _spec(str(tmp_path / "out"), step=oom_step), batch_size=4)
    assert report.batch_shapes == {2: 6, 1: 1}
    assert report.oom_events == [{"at": "img00.png", "from": 4, "to": 2}]
    assert report.summary()["oom_events"] == report.oom_events


def test_out_of_memory_at_size_one_fails_only_that_item():
    backoff = OomBackoff(4)

    def step(x):
        if x.shape[0] > 1 or float(x[0, 0]) == 2.0:
            raise torch.cuda.OutOfMemoryError("stub")
        return x.clone()

    results = backoff.run(step, torch.arange(4.0).reshape(4, 1))
    ok = [(s.start, out) for s, out in results if not isinstance(out, BaseException)]
    bad = [s.start for s, out in results if isinstance(out, BaseException)]
    assert bad == [2]
    assert [start for start, _ in ok] == [0, 1, 3]
    assert backoff.size == 1


def test_failed_batch_memory_is_released_before_the_retry():
    """The OOM traceback pins the failed forward's activations; free them first."""
    import gc
    import weakref

    refs = []
    alive_at_retry = []

    def step(x):
        if x.shape[0] > 1:
            activation = torch.zeros(1000)
            refs.append(weakref.ref(activation))
            raise torch.cuda.OutOfMemoryError("stub")
        return x.clone()

    def on_oom():
        gc.collect()
        alive_at_retry.append(any(r() is not None for r in refs))

    results = OomBackoff(2, on_oom=on_oom).run(step, torch.zeros(2, 1))
    assert alive_at_retry == [False]
    assert len(results) == 2


def test_single_item_failure_carries_no_traceback():
    def step(x):
        raise torch.cuda.OutOfMemoryError("stub")

    [(chunk, result)] = OomBackoff(1).run(step, torch.zeros(1, 1))
    assert isinstance(result, BaseException) and result.__traceback__ is None
    assert "stub" in str(result)


def test_non_oom_errors_are_not_retried():
    backoff = OomBackoff(4)

    def step(x):
        raise ValueError("bad model")

    with pytest.raises(ValueError):
        backoff.run(step, torch.zeros(4, 1))


def test_images_out_of_memory_alone_log_the_first_few_then_a_count(tmp_path, caplog):
    paths = _make_images(str(tmp_path / "in"), [f"i{i:02d}.png" for i in range(25)])

    def oom_step(x):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory (stub)")

    with caplog.at_level(logging.WARNING, logger=ENGINE_LOGGER):
        report = run_batches(paths, _spec(str(tmp_path / "out"), step=oom_step), batch_size=4)
    assert len(report.failed) == 25 and report.written == []
    assert len(_messages(caplog, "Out of memory on a single image")) == FAILURE_LOG_LIMIT
    [more] = _messages(caplog, "more single images out of GPU memory")
    assert more.getMessage().startswith("5 more")


class _FakeOom(Exception):
    """Stands in for ``torch.cuda.OutOfMemoryError`` (injected through ``is_oom``)."""


def _backoff(batch_size):
    return OomBackoff(batch_size, is_oom=lambda exc: isinstance(exc, _FakeOom), on_oom=lambda: None)


def _step_fitting(limit, attempts):
    """A step that runs out of memory on more than ``limit`` items; records each attempt.

    It also fails the test when a slice of more than one item is tried although a slice
    at most as long already ran out of memory (a wasted retry).
    """
    shortest_oom = float("inf")

    def step(batch):
        nonlocal shortest_oom
        attempts.append(len(batch))
        assert len(batch) == 1 or len(batch) < shortest_oom, f"retried {len(batch)} items"
        if len(batch) > limit:
            shortest_oom = min(shortest_oom, len(batch))
            raise _FakeOom(f"{len(batch)} items do not fit")
        return list(batch)

    return step


def _legacy_halving(n, limit):
    """What the pre-R5 halving did with ``n`` items when ``limit`` of them fit at a time.

    Returns ``(slices that ran, items that failed alone, final size)``.
    """
    ran, failed, size = [], [], n

    def visit(start, stop):
        nonlocal size
        if stop - start <= limit:
            ran.append((start, stop))
        elif stop - start == 1:
            failed.append(start)
        else:
            half = (stop - start + 1) // 2
            size = min(size, half)
            visit(start, start + half)
            visit(start + half, stop)

    visit(0, n)
    return ran, failed, size


def test_the_sibling_half_is_not_retried_at_a_size_that_ran_out_of_memory():
    attempts = []
    backoff = _backoff(16)
    results = backoff.run(_step_fitting(4, attempts), list(range(16)))

    assert attempts == [16, 8, 4, 4, 4, 4]  # before R5: [16, 8, 4, 4, 8, 4, 4]
    assert [(s.start, s.stop) for s, _ in results] == [(0, 4), (4, 8), (8, 12), (12, 16)]
    assert backoff.size == 4


def test_a_shorter_sibling_half_is_still_tried_so_the_old_slices_run():
    attempts = []
    results = _backoff(7).run(_step_fitting(3, attempts), list(range(7)))

    assert [s.stop - s.start for s, _ in results] == [2, 2, 3]  # as before R5
    assert attempts == [7, 4, 2, 2, 3]  # 3 is shorter than the 4 that ran out of memory


def test_an_odd_batch_covers_every_item_once_in_order():
    attempts = []
    results = _backoff(7).run(_step_fitting(2, attempts), list(range(7)))

    assert [(s.start, s.stop) for s, _ in results] == [(0, 2), (2, 4), (4, 6), (6, 7)]
    assert [item for _, items in results for item in items] == list(range(7))
    assert attempts == [7, 4, 2, 2, 3, 2, 1]


@pytest.mark.parametrize("n", range(1, 41))
def test_under_a_fixed_threshold_the_old_halving_runs_unchanged(n):
    for limit in range(n + 1):  # 0: not even a single item fits
        backoff = _backoff(n)
        results = backoff.run(_step_fitting(limit, []), list(range(n)))

        ran = [(s.start, s.stop) for s, out in results if not isinstance(out, BaseException)]
        failed = [s.start for s, out in results if isinstance(out, BaseException)]
        assert [i for s, _ in results for i in range(s.start, s.stop)] == list(range(n)), limit
        assert (ran, failed, backoff.size) == _legacy_halving(n, limit), limit


def test_a_lone_image_out_of_memory_is_recorded_once_with_the_original_reason(tmp_path):
    paths = _make_images(str(tmp_path / "in"), ["a.png", "b.png", "c.png"])

    def step(x):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory (stub)")

    report = run_batches(paths, _spec(str(tmp_path / "out"), step=step), batch_size=2)

    reason = "OutOfMemoryError: CUDA out of memory (stub)"  # not "ItemOutOfMemory: ..."
    assert report.failed == [("a.png", reason), ("b.png", reason), ("c.png", reason)]
    assert report.written == []
    assert report.effective_batch_size == 1


def _recording(sizes, step=_step):
    def recording_step(x):
        sizes.append(x.shape[0])
        return step(x)

    return recording_step


def test_a_padded_tail_runs_every_forward_at_the_fixed_size(tmp_path):
    paths = _make_images(str(tmp_path / "in"), NAMES)
    sizes = []
    out = str(tmp_path / "out")

    report = run_batches(paths, _spec(out, step=_recording(sizes)), batch_size=3, fixed=True)
    ref = str(tmp_path / "ref")
    run_batches(paths, _spec(ref), batch_size=3, fixed=False)

    assert sizes == [3, 3, 3]
    assert len(report.written) == 7
    assert _outputs(out) == _outputs(ref)
    assert report.batch_shapes == {3: 7}
    assert report.effective_batch_size == 3
    assert report.oom_events == []


def test_padding_rows_are_never_saved(tmp_path):
    paths = _make_images(str(tmp_path / "in"), NAMES)
    saved = []

    def save(arr, path):
        saved.append(os.path.basename(path))

    run_batches(paths, _spec(str(tmp_path / "out"), save=save), batch_size=3, fixed=True)
    assert sorted(saved) == NAMES


def test_an_unreadable_input_keeps_its_slot_and_the_tail_is_padded(tmp_path):
    dirty = _make_images(str(tmp_path / "in"), NAMES + ["bad.jpg"], corrupt={"bad.jpg"})
    sizes = []

    report = run_batches(
        dirty, _spec(str(tmp_path / "out"), step=_recording(sizes)), batch_size=3, fixed=True
    )

    assert sizes == [3, 3, 3]
    assert report.failed_names == ["bad.jpg"]
    assert report.batch_shapes == {3: 8}


def test_auto_keeps_the_short_tail(tmp_path):
    paths = _make_images(str(tmp_path / "in"), NAMES)
    sizes = []
    run_batches(paths, _spec(str(tmp_path / "out"), step=_recording(sizes)), batch_size=3)
    assert sizes == [3, 3, 1]


def test_fixed_out_of_memory_stops_without_a_retry(tmp_path, monkeypatch):
    paths = _make_images(str(tmp_path / "in"), NAMES)
    sizes, frees = [], []
    monkeypatch.setattr(batch_engine, "_free_cuda_cache", lambda: frees.append(1))

    def oom_step(x):
        sizes.append(x.shape[0])
        raise torch.cuda.OutOfMemoryError("CUDA out of memory (stub)")

    report = BatchReport()
    with pytest.raises(FixedBatchOutOfMemory) as caught:
        run_batches(
            paths,
            _spec(str(tmp_path / "out"), step=oom_step),
            batch_size=3,
            fixed=True,
            report=report,
        )

    assert str(caught.value) == FIXED_OOM_MESSAGE.format(n=3)
    assert caught.value.batch_size == 3
    assert sizes == [3]  # no retry, no halving
    assert frees == [1]
    assert report.oom_events == []
    assert report.written == []
    assert report.effective_batch_size == 3


def test_the_out_of_memory_message_names_the_size_and_the_way_out():
    text = str(FixedBatchOutOfMemory(4))
    assert text.startswith("Out of GPU memory at the fixed batch size (4). ")
    assert "SPROUTS_BATCH_POLICY=auto" in text


def test_a_non_memory_error_still_propagates_unchanged_when_fixed(tmp_path):
    paths = _make_images(str(tmp_path / "in"), NAMES)
    error = ValueError("bad model")

    def step(x):
        raise error

    with pytest.raises(ValueError) as caught:
        run_batches(paths, _spec(str(tmp_path / "out"), step=step), batch_size=3, fixed=True)
    assert caught.value is error
