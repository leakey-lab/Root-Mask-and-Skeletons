"""``metrics.run_metric_pool``: the Measure thread pool.

The rows, their order and the progress values are those of measuring the images one by one
(ties by image name, the same on every OS); a stop is heard while tasks still measure and the
rows so far are returned; a worker's error ends the pool without measuring the rest; a stopped
run frees its tasks by refcount (no cycle a ``gc.freeze()`` would keep).
"""

from __future__ import annotations

import pytest

from app.inference import metrics


def test_run_metric_pool_stops_early_when_asked():
    calls = []

    def worker(name, path):
        calls.append(name)
        return {"Image": name}

    images = {str(i): "p" for i in range(400)}
    rows = metrics.run_metric_pool(images, worker, should_stop=lambda: len(calls) >= 5)
    assert len(rows) < len(images)


@pytest.mark.wallclock
def test_a_stop_is_heard_while_every_task_is_still_measuring():
    """``should_stop`` is asked while no task ends, not only when one does: a Cancel never
    waits for a whole run of images (up to 32 per task)."""
    import threading
    import time

    release = threading.Event()

    def worker(name, path):
        release.wait(0.5)  # each image is slow until the stop was asked for
        return {"Image": name}

    def should_stop():
        release.set()
        return True

    images = {f"{i:04d}": "p" for i in range(4096)}  # tasks of 32 images: 16 s each
    started = time.perf_counter()
    rows = metrics.run_metric_pool(images, worker, should_stop=should_stop)
    assert time.perf_counter() - started < 5
    assert len(rows) < 256


def test_a_stopped_run_frees_its_tasks_by_refcount(monkeypatch):
    """A task still in the queue of finished tasks when a stopped run returns refers to that
    queue through its done callback: a cycle, which a ``gc.freeze()`` (``app.gc_policy``,
    after a warm-up) before the next full collection would keep for the rest of the session."""
    import threading
    import weakref
    from concurrent.futures import ThreadPoolExecutor

    from tests.leak_census import assert_freed, refcount_only

    tasks = []

    class Recording(ThreadPoolExecutor):
        def submit(self, *args, **kwargs):
            future = super().submit(*args, **kwargs)
            tasks.append(weakref.ref(future))
            return future

    monkeypatch.setattr(metrics, "ThreadPoolExecutor", Recording)
    lock = threading.Lock()
    measured = [0]

    def worker(name, path):
        with lock:
            measured[0] += 1
        return {"Image": name}

    images = {f"{i:04d}": "p" for i in range(3000)}
    with refcount_only():
        rows = metrics.run_metric_pool(images, worker, should_stop=lambda: measured[0] >= 600)
        assert len(rows) < len(images)  # stopped with tasks left over
        del rows
        assert len(tasks) > 1
        for ref in tasks:
            assert_freed(ref, "a task of the stopped run")


def test_an_error_in_a_worker_ends_the_pool_without_measuring_the_rest():
    """The error is raised once the running images end: the pending ones are not measured
    first (a Cancel meanwhile would go unheard), and it is the worker's own error."""
    import time

    calls = []

    def worker(name, path):
        calls.append(name)
        if name == "0":
            raise MemoryError("simulated")
        time.sleep(0.005)
        return {"Image": name}

    images = {str(i): "p" for i in range(400)}
    with pytest.raises(MemoryError, match="simulated"):
        metrics.run_metric_pool(images, worker)
    assert len(calls) < len(images) // 2


@pytest.mark.parametrize("total", [5, 255, 256, 1000, 4100])
def test_the_pool_gives_the_rows_and_progress_of_one_image_at_a_time(total):
    """Images run in tasks of several (fewer futures); the rows, their order and the
    progress values are those of measuring the images one by one."""
    import random
    import time

    def worker(name, path):
        if random.random() < 0.01:
            time.sleep(0.001)
        return {"Image": name, "Tube": int(name) % 7 or None, "Path": path}

    images = {f"{i:05d}": f"p{i}" for i in random.Random(total).sample(range(10 * total), total)}
    progress = []
    rows = metrics.run_metric_pool(images, worker, progress_cb=progress.append)
    serial = sorted((worker(name, path) for name, path in images.items()), key=metrics._sort_key)
    assert rows == serial
    expected, last = [], -1
    for done in range(1, total + 1):
        pct = int(done / total * 100)
        if pct != last:
            expected.append(pct)
            last = pct
    assert progress == expected


def test_rows_with_equal_sort_keys_come_out_by_image_name():
    """Tied rows (no metadata, error rows) are ordered by image name, not by folder-scan
    order (which differs per OS) or thread completion (WX-T9)."""
    import random
    import time

    def worker(name, path):
        time.sleep(random.random() / 500)
        return {"Image": name}

    images = {name: "" for name in ("img07", "img02", "img11", "IMG_0001")}
    for _ in range(3):
        rows = metrics.run_metric_pool(images, worker)
        assert [r["Image"] for r in rows] == sorted(images)


class _BrokenProgress:
    """A progress callback that raises on its first ``fail_first`` calls."""

    def __init__(self, fail_first=1):
        self.fail_first = fail_first
        self.calls = []

    def __call__(self, *args):
        self.calls.append(args)
        if len(self.calls) <= self.fail_first:
            raise RuntimeError("progress listener is gone (simulated)")


# R3: a progress callback that raised failed the run after every image had been measured.
def test_a_raising_progress_callback_never_fails_a_measure_run(tmp_path):
    images = {f"img{i}": str(tmp_path / f"img{i}.png") for i in range(4)}
    progress = _BrokenProgress()

    rows = metrics.run_metric_pool(
        images, lambda name, path: {"Image": name, "Tube": None}, progress_cb=progress
    )

    assert sorted(row["Image"] for row in rows) == sorted(images)
    assert progress.calls[-1] == (100,)  # progress still reaches 100 %
