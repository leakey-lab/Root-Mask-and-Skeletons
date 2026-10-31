"""R7: a full disk ends a generation run at the next batch boundary.

A failed save is recorded per image and the run goes on (PI-03): right for a mask
another program holds open, wrong for a full disk. Once the disk is full every later
save fails too, but the run went on decoding and running the model over the rest of
the folder (for 46k images, minutes of GPU time) only to report thousands of failures
at the end. A save that fails with ENOSPC now stops the run at the next batch
boundary with that error: the outputs already on disk are kept and listed, the run
manifest says "failed" (R7), and the handler shows the message at once.

C6-01: an output folder that takes no new files (permission denied, a read-only drive, a
folder that is gone) stops the run the same way, with one message naming the folder.
"""

from __future__ import annotations

import errno
import os
import threading

import pytest

from app.inference.batch_engine import BatchReport, run_batches
from tests.test_generation_batch_engine import NAMES, _make_images, _spec, _step


def _disk_fills_after(n_saves, out_dir, *, err=errno.ENOSPC):
    """A save that writes the first ``n_saves`` files, then fails like a full disk."""
    from PIL import Image

    saved = []

    def save(arr, path):
        if len(saved) >= n_saves:
            raise OSError(err, os.strerror(err), path)
        saved.append(path)
        stem = os.path.splitext(os.path.basename(path))[0]
        Image.fromarray(arr, mode="L").save(os.path.join(out_dir, stem + ".png"))

    return save


def test_a_full_disk_stops_the_run_at_the_next_batch(tmp_path):
    paths = _make_images(str(tmp_path / "in"), NAMES)  # 7 images
    out = str(tmp_path / "out")
    steps = []
    full = threading.Event()
    fill = _disk_fills_after(2, out)

    def save(arr, path):
        try:
            return fill(arr, path)
        except OSError:
            full.set()
            raise

    def step(x):
        # The writer runs on its own thread: the third batch waits until the disk is
        # full, so the stop is checked at the next boundary whatever the timing.
        if len(steps) == 2:
            full.wait(5)
        steps.append(len(x))
        return _step(x)

    report = BatchReport()
    with pytest.raises(OSError) as caught:
        run_batches(
            paths,
            _spec(out, step=step, save=save),
            batch_size=2,
            io_threads=1,
            report=report,
        )

    assert caught.value.errno == errno.ENOSPC
    assert report.written == sorted(os.listdir(out)) == NAMES[:2]  # what reached the disk
    assert report.failed and all(
        "No space" in reason or "28" in reason for _, reason in report.failed
    )
    assert len(steps) < 4  # the model did not run over the whole folder


def test_other_save_errors_still_fail_one_image_each(tmp_path):
    paths = _make_images(str(tmp_path / "in"), NAMES)
    out = str(tmp_path / "out")

    report = run_batches(
        paths,
        _spec(out, save=_disk_fills_after(2, out, err=errno.EACCES)),
        batch_size=2,
        io_threads=1,
    )

    assert report.status == "completed"
    assert len(report.written) == 2 and len(report.failed) == len(NAMES) - 2


# --------------------------------------------------------------------------- C6-01


def _atomic_save(out_dir):
    """The generation save shape: an atomic bulk write into ``out_dir``."""
    from app.io_utils import atomic_write_bytes

    def save(arr, path):
        stem = os.path.splitext(os.path.basename(path))[0]
        atomic_write_bytes(os.path.join(out_dir, stem + ".png"), arr.tobytes(), durable=False)

    return save


def _run_until_stopped(paths, out, save, report):
    """Run 7 images in batches of 2; the third batch waits until the first save has failed,
    so the stop is checked at the next boundary whatever the timing. Returns the forwards'
    sizes. The wait is on progress, which counts a save once the engine has handled it."""
    steps = []
    one_saved = threading.Event()

    def step(x):
        if len(steps) == 2:
            one_saved.wait(5)
        steps.append(len(x))
        return _step(x)

    with pytest.raises(OSError) as caught:
        run_batches(
            paths,
            _spec(out, step=step, save=save),
            batch_size=2,
            io_threads=1,
            progress=lambda done, total: one_saved.set(),
            report=report,
        )
    return steps, caught.value


def test_an_output_folder_that_refuses_files_stops_the_run_naming_it(tmp_path, monkeypatch):
    """C6-01: a folder that takes no new files (here: permission denied) ends the run at the
    next batch boundary with one message naming the folder, as a full disk does. It used to
    run the model over every image (about 1-1.5 h at 46k) and then list each one as failed."""
    from app import io_utils

    paths = _make_images(str(tmp_path / "in"), NAMES)
    out = str(tmp_path / "out")
    real_open = open

    def refusing_open(path, mode="r", *args, **kwargs):
        if "x" in mode:  # the temp file of an atomic save
            raise PermissionError(errno.EACCES, "Permission denied", path)
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(io_utils, "open", refusing_open, raising=False)
    report = BatchReport()
    steps, err = _run_until_stopped(paths, out, _atomic_save(out), report)

    assert isinstance(err, io_utils.FolderNotWritableError)
    assert err.folder == os.path.abspath(out)
    assert str(err) == (
        f"Cannot save files in {os.path.abspath(out)}: Permission denied. "
        "Check that the folder exists and that you may write to it, then try again."
    )
    assert isinstance(err.__cause__, PermissionError)
    assert len(steps) < 4  # the model did not run over the whole folder
    assert report.written == []
    # Each failed save names its own file, never the hidden temp file (C6-05).
    assert report.failed and all(".tmp" not in reason for _, reason in report.failed)
    assert all("img" in reason for _, reason in report.failed)


def test_an_output_folder_that_is_gone_stops_the_run_naming_it(tmp_path):
    """The real thing, on every platform: the output folder was removed (or its drive
    unplugged) before the saves."""
    from app.io_utils import FolderNotWritableError

    paths = _make_images(str(tmp_path / "in"), NAMES)
    out = str(tmp_path / "out")
    spec_dir = str(tmp_path / "spec")  # _spec makes its folder; the saves go to ``out``
    report = BatchReport()
    steps, err = _run_until_stopped(paths, spec_dir, _atomic_save(out), report)

    assert isinstance(err, FolderNotWritableError)
    assert err.folder == os.path.abspath(out)
    assert os.path.abspath(out) in str(err)
    assert len(steps) < 4
    assert report.written == [] and report.failed


def test_no_such_file_for_one_name_in_an_existing_folder_fails_one_image(tmp_path, monkeypatch):
    """Windows says ENOENT for one output path over MAX_PATH (long paths off) while the
    folder is there. That image fails; the others are saved and the run completes."""
    from app import io_utils

    paths = _make_images(str(tmp_path / "in"), NAMES)
    out = str(tmp_path / "out")
    os.makedirs(out)
    long_one = os.path.splitext(NAMES[0])[0]
    real_open = open

    def open_too_long(path, mode="r", *args, **kwargs):
        if "x" in mode and os.path.basename(path).startswith("." + long_one):
            raise FileNotFoundError(errno.ENOENT, "No such file or directory", path)
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(io_utils, "open", open_too_long, raising=False)
    report = run_batches(paths, _spec(out, save=_atomic_save(out)), batch_size=2, io_threads=1)

    assert report.status == "completed"
    assert [name for name, _ in report.failed] == [NAMES[0]]
    assert ".tmp" not in report.failed[0][1]  # (C6-05)
    assert sorted(report.written) == sorted(os.listdir(out)) == NAMES[1:]


def test_a_save_that_fails_for_its_own_file_still_fails_one_image(tmp_path, monkeypatch):
    """A swap refused for one file (a read-only output, say) says nothing of the folder."""
    from app import io_utils

    paths = _make_images(str(tmp_path / "in"), NAMES)
    out = str(tmp_path / "out")
    os.makedirs(out)

    def refuse(src, dst):
        raise PermissionError(errno.EACCES, "Permission denied", src)

    monkeypatch.setattr(io_utils.os, "replace", refuse)
    report = run_batches(paths, _spec(out, save=_atomic_save(out)), batch_size=2, io_threads=1)

    assert report.status == "completed"
    assert len(report.failed) == len(NAMES)
    assert all(".tmp" not in reason for _, reason in report.failed)  # (C6-05)
    assert os.listdir(out) == []  # no temp file left behind
