"""Stale temp files of killed writes are swept (R8).

``app.io_utils.atomic_open`` writes to a hidden ``.<name>.<8 hex>.tmp`` next to the
target and swaps it in. A process killed in between left that file behind for good
(the review's repro: ``.root_lengths.csv.78684f58.tmp``, 385 KB). Opening a project
now sweeps its output folders for such files older than an hour, off the GUI thread.

The per-user settings folder (the Enhance defaults, ``normalization_defaults.json``) is swept
too, by ``EnhanceDefaultsStore.load()``: once per process, on a daemon thread, with the same
age rule; ``load()`` never waits for it and never raises, even when no thread can start (R4).

The GT store and registry had their own non-durable writers (``<path>.tmp``); their old
``<name>.tmp`` leftovers are swept from ``gt/``. (What those writers write today:
``tests/test_gt_writes.py``.)
"""

from __future__ import annotations

import errno
import json
import logging
import os
import threading
import time

import numpy as np
import pytest

from app import enhancement, io_utils
from app.enhancement import DEFAULTS_FILE_NAME, EnhanceDefaultsStore
from app.io_utils import atomic_open
from app.project_layout import ProjectLayout

HOUR = 3600.0
STEM = "24EFCAM1_T001_L045_2024.07.18_111618_001_SMM"


def sweep(folder, **kwargs):
    return io_utils.sweep_stale_temp_files(folder, **kwargs)


def _file(folder, name: str, *, age_s: float = 0.0, text: str | None = None) -> str:
    """Create ``folder/name`` whose mtime is ``age_s`` seconds in the past; its content is
    ``text`` (UTF-8) when given, else the bytes of a partial write."""
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(str(folder), name)
    with open(path, "wb") as fh:
        fh.write(b"partial write" if text is None else text.encode("utf-8"))
    if age_s:
        stamp = time.time() - age_s
        os.utime(path, (stamp, stamp))
    return path


# ------------------------------------------------------------------ the sweeper


def test_a_stale_temp_file_is_removed_and_a_fresh_one_is_kept(tmp_path):
    stale = _file(tmp_path, ".root_lengths.csv.78684f58.tmp", age_s=2 * HOUR)
    fresh = _file(tmp_path, ".root_areas.csv.0badf00d.tmp", age_s=60)  # a write in progress
    target = _file(tmp_path, "root_lengths.csv", age_s=2 * HOUR)

    assert sweep(tmp_path) == [stale]
    assert sorted(os.listdir(tmp_path)) == sorted(map(os.path.basename, (fresh, target)))


def test_the_age_threshold_can_be_chosen(tmp_path):
    older = _file(tmp_path, ".a.png.00000001.tmp", age_s=90)
    younger = _file(tmp_path, ".b.png.00000002.tmp", age_s=30)

    assert sweep(str(tmp_path), older_than_s=60) == [older]
    assert os.path.exists(younger)


@pytest.mark.parametrize(
    "name",
    [
        "notes.tmp",  # not hidden
        ".hidden.txt",  # not .tmp
        "x.png.1a2b3c4d.tmp",  # no leading dot
        ".x.ABCDEF12.tmp",  # upper-case hex: token_hex never makes it
        ".x.1a2b3c4.tmp",  # 7 hex digits
        ".x.1a2b3c4d5.tmp",  # 9 hex digits
        ".x.1a2b3c4g.tmp",  # not hex
        ".1a2b3c4d.tmp",  # no target name
        ".x.png.1a2b3c4d.tmp.bak",
        "classes.json.tmp",  # a legacy GT name: swept in gt/ only
        f"{STEM}.png.tmp",
    ],
)
def test_names_that_are_not_atomic_temp_files_are_kept(tmp_path, name):
    path = _file(tmp_path, name, age_s=2 * HOUR)
    assert sweep(tmp_path) == []
    assert os.path.exists(path)


def test_a_folder_named_like_a_temp_file_is_kept(tmp_path, caplog):
    folder = tmp_path / ".mask.png.1a2b3c4d.tmp"
    folder.mkdir()
    stamp = time.time() - 2 * HOUR
    os.utime(folder, (stamp, stamp))
    with caplog.at_level(logging.WARNING, logger=io_utils.__name__):
        assert sweep(tmp_path) == []
    assert folder.is_dir()
    assert caplog.records == []  # skipped, not a failed delete on every project load


def test_the_sweep_never_recurses(tmp_path):
    nested = _file(tmp_path / "sub", ".a.png.1a2b3c4d.tmp", age_s=2 * HOUR)
    assert sweep(tmp_path) == []
    assert os.path.exists(nested)


def test_a_missing_folder_or_a_file_gives_an_empty_list(tmp_path):
    assert sweep(tmp_path / "missing") == []
    assert sweep(_file(tmp_path, "a.png")) == []


def test_an_unreadable_folder_is_logged_not_raised(tmp_path, monkeypatch, caplog):
    def denied(path):
        raise PermissionError(errno.EACCES, "Access is denied", str(path))

    monkeypatch.setattr(io_utils.os, "scandir", denied)
    with caplog.at_level(logging.WARNING, logger=io_utils.__name__):
        assert sweep(tmp_path) == []
    assert str(tmp_path) in caplog.text


def test_a_file_that_cannot_be_deleted_is_logged_and_skipped(tmp_path, monkeypatch, caplog):
    held = _file(tmp_path, ".a.png.1a2b3c4d.tmp", age_s=2 * HOUR)
    other = _file(tmp_path, ".b.png.1a2b3c4e.tmp", age_s=2 * HOUR)
    real_remove = os.remove

    def remove(path):
        if os.path.basename(path) == os.path.basename(held):
            raise PermissionError(errno.EACCES, "The process cannot access the file", path)
        real_remove(path)

    monkeypatch.setattr(io_utils.os, "remove", remove)
    with caplog.at_level(logging.WARNING, logger=io_utils.__name__):
        assert sweep(tmp_path) == [other]
    assert os.path.exists(held) and not os.path.exists(other)
    assert os.path.basename(held) in caplog.text


@pytest.mark.parametrize(
    "target", ["root_lengths.csv", f"{STEM}_fake.png", ".hidden", "Données_根.png", "x.tmp"]
)
def test_what_a_killed_atomic_write_leaves_is_swept(tmp_path, monkeypatch, target):
    # Keep atomic_open from cleaning up, as a killed process would.
    monkeypatch.setattr(io_utils, "_remove_quietly", lambda path: None)
    with pytest.raises(RuntimeError), atomic_open(tmp_path / target, "wb") as fh:
        fh.write(b"half a file")
        raise RuntimeError("killed mid-write")
    (leftover,) = os.listdir(tmp_path)
    stamp = time.time() - 2 * HOUR
    os.utime(tmp_path / leftover, (stamp, stamp))

    assert sweep(tmp_path) == [os.path.join(str(tmp_path), leftover)]
    assert os.listdir(tmp_path) == []


# ------------------------------------------------------------------ project folders


def test_output_dirs_are_the_folders_the_app_writes_into(tmp_path):
    root = str(tmp_path)
    join = os.path.join
    # The older layout (output/mask, [output/]skeletonizer/test_latest/images) is retired (#38).
    assert ProjectLayout(root).output_dirs == (
        join(root, "mask"),
        join(root, "skeletons"),
        join(root, "gt"),
        join(root, "gt", "mask"),
        join(root, "gt", "skeletons"),
    )
    assert os.listdir(tmp_path) == []  # describing the folders creates none


def test_output_dirs_hold_the_folders_measure_writes_its_csvs_into(tmp_path):
    # Measure writes root_areas.csv / root_lengths.csv (+ .provenance.json) next to the
    # predictions it reads, in mask/ and skeletons/ (skeleton_handler).
    layout = ProjectLayout(str(tmp_path))
    _file(layout.mask_dir, "root_areas.csv")
    _file(layout.skeletons_dir, "root_lengths.csv")
    assert os.path.dirname(layout.find_areas_csv()) in layout.output_dirs
    assert os.path.dirname(layout.find_lengths_csv()) in layout.output_dirs


@pytest.mark.parametrize(
    ("name", "legacy"),
    [
        ("classes.json.tmp", True),
        (f"{STEM}.png.tmp", True),
        ("a.b.png.tmp", True),
        ("notes.tmp", False),
        ("classes.json", False),
        (f"{STEM}.png", False),
        ("a.jpg.tmp", False),
        ("other.json.tmp", False),
        (f"{STEM}.png.tmp.bak", False),
    ],
)
def test_legacy_gt_temp_names(name, legacy):
    # gt/store.py and gt/registry.py wrote "<path>.tmp" before R8.
    assert ProjectLayout.is_legacy_gt_temp(name) is legacy


def _stale_project(root):
    """A project with stale and fresh temp files in every output folder."""
    layout = ProjectLayout(str(root))
    stale, kept = set(), set()
    for folder in layout.output_dirs:
        stale.add(_file(folder, ".x.png.1a2b3c4d.tmp", age_s=2 * HOUR))
        kept.add(_file(folder, ".y.png.1a2b3c4e.tmp", age_s=60))
        kept.add(_file(folder, "notes.tmp", age_s=2 * HOUR))
    stale.add(_file(layout.gt_dir, "classes.json.tmp", age_s=2 * HOUR))
    stale.add(_file(layout.gt_dir, f"{STEM}.png.tmp", age_s=2 * HOUR))
    kept.add(_file(layout.gt_dir, "a.png.tmp", age_s=60))  # legacy but fresh
    kept.add(_file(layout.mask_dir, f"{STEM}.png.tmp", age_s=2 * HOUR))  # legacy: gt/ only
    kept.add(_file(layout.gt_skeletons_dir, "classes.json.tmp", age_s=2 * HOUR))
    return layout, stale, kept


def test_the_project_sweep_covers_every_output_folder(tmp_path):
    from app.gui.image_manager import sweep_project_temp_files

    layout, stale, kept = _stale_project(tmp_path)

    assert set(sweep_project_temp_files(layout)) == stale
    assert not any(os.path.exists(p) for p in stale)
    assert all(os.path.exists(p) for p in kept)


@pytest.mark.wallclock
def test_a_loaded_project_is_swept_on_a_daemon_thread_after_its_scan(app, tmp_path, monkeypatch):
    import cv2

    from app.gui import image_manager as im

    layout, stale, kept = _stale_project(tmp_path)
    cv2.imwrite(os.path.join(layout.root, f"{STEM}.png"), np.zeros((8, 8, 3), np.uint8))
    release, calls, scheduled = threading.Event(), [], []
    real_sweep = im.sweep_project_temp_files

    def held_sweep(swept_layout):
        if swept_layout != layout:  # a delayed sweep scheduled by an earlier test's window
            return real_sweep(swept_layout)
        calls.append((swept_layout, threading.current_thread()))
        release.wait(10)
        return real_sweep(swept_layout)

    class _Timer:  # records the deferred start instead of waiting for it
        @staticmethod
        def singleShot(ms, fn):  # noqa: N802 - Qt API name
            scheduled.append((ms, fn))

    monkeypatch.setattr(im, "sweep_project_temp_files", held_sweep)
    monkeypatch.setattr(im, "QTimer", _Timer)
    manager = im.ImageManager(None)
    started = time.monotonic()
    manager.load_images(layout.root)
    returned_after = time.monotonic() - started
    try:
        # Nothing sweeps while the library scans the same folders: the start is
        # scheduled once the scan has delivered, TEMP_SWEEP_DELAY_MS later.
        deadline = time.monotonic() + 10
        while not scheduled and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.01)
        assert manager.original_folder == layout.root
        ((delay, start),) = scheduled
        assert delay == im.TEMP_SWEEP_DELAY_MS > 0
        assert calls == []
        start()
        deadline = time.monotonic() + 5
        while not calls and time.monotonic() < deadline:
            time.sleep(0.01)
        stale_while_held = [os.path.exists(p) for p in stale]
    finally:
        release.set()
        # The manager lets a scan's thread go once it has ended.
        if manager.loader_thread is not None:
            manager.loader_thread.wait(5000)

    assert returned_after < 2.0  # load_images never waits for the sweep
    ((swept_layout, thread),) = calls
    assert swept_layout == layout
    assert thread is not threading.main_thread() and thread.daemon
    assert all(stale_while_held)
    thread.join(5)
    assert not any(os.path.exists(p) for p in stale)
    assert all(os.path.exists(p) for p in kept)


# ------------------------------------------------------------------ the settings folder (R4)


@pytest.fixture(autouse=True)
def _fresh_sweep_state(monkeypatch):
    # Each test sees a process in which no settings folder was swept yet.
    monkeypatch.setattr(enhancement, "_swept_settings_dirs", set())


def _store(folder):
    factory = _file(
        folder.parent / "factory", DEFAULTS_FILE_NAME, text=json.dumps({"clahe_tile": 9})
    )
    return EnhanceDefaultsStore(user_path=str(folder / DEFAULTS_FILE_NAME), factory_path=factory)


def test_loading_the_defaults_sweeps_stale_temp_files_from_the_settings_folder(tmp_path):
    folder = tmp_path / "SPROUTS"
    stale = _file(folder, f".{DEFAULTS_FILE_NAME}.1a2b3c4d.tmp", age_s=2 * HOUR)
    fresh = _file(folder, f".{DEFAULTS_FILE_NAME}.0badf00d.tmp", age_s=60)  # a save in progress
    other = _file(folder, "notes.tmp", age_s=2 * HOUR)  # not an atomic-write temp name
    store = _store(folder)

    assert store.load().tile == 9  # seeded from the factory file, as before
    store.sweep_thread.join(5)

    assert not os.path.exists(stale)
    assert os.path.exists(fresh) and os.path.exists(other)
    assert os.path.exists(store.user_path)


@pytest.mark.wallclock
def test_the_sweep_runs_on_a_daemon_thread_and_load_never_waits_for_it(tmp_path, monkeypatch):
    folder = tmp_path / "SPROUTS"
    stale = _file(folder, f".{DEFAULTS_FILE_NAME}.1a2b3c4d.tmp", age_s=2 * HOUR)
    release, calls = threading.Event(), []
    real_sweep = enhancement.sweep_stale_temp_files

    def held_sweep(swept_folder, **kwargs):
        calls.append((swept_folder, threading.current_thread()))
        release.wait(10)
        return real_sweep(swept_folder, **kwargs)

    monkeypatch.setattr(enhancement, "sweep_stale_temp_files", held_sweep)
    store = _store(folder)
    started = time.monotonic()
    try:
        store.load()
        returned_after = time.monotonic() - started
        deadline = time.monotonic() + 5
        while not calls and time.monotonic() < deadline:
            time.sleep(0.01)
        assert os.path.exists(stale)  # still held: load() did not wait
    finally:
        release.set()
    store.sweep_thread.join(5)

    assert returned_after < 2.0
    ((swept_folder, thread),) = calls
    assert os.path.normcase(swept_folder) == os.path.normcase(str(folder))
    assert thread is not threading.main_thread() and thread.daemon
    assert not os.path.exists(stale)


def test_the_settings_folder_is_swept_once_per_process(tmp_path, monkeypatch):
    folder = tmp_path / "SPROUTS"
    calls = []
    monkeypatch.setattr(
        enhancement, "sweep_stale_temp_files", lambda f, **kw: calls.append(f) or []
    )

    first = _store(folder)
    first.load()
    first.sweep_thread.join(5)
    second = _store(folder)  # e.g. the Trace GT editor's controls after Correct's
    second.load()
    first.load()

    assert len(calls) == 1
    assert second.sweep_thread is None


def test_a_missing_settings_folder_is_not_an_error(tmp_path):
    store = EnhanceDefaultsStore(
        user_path=str(tmp_path / "missing" / DEFAULTS_FILE_NAME),
        factory_path=str(tmp_path / "no_factory.json"),
    )
    assert store.load() == enhancement.DEFAULT_SETTINGS
    store.sweep_thread.join(5)
    assert not (tmp_path / "missing").exists()  # sweeping creates nothing


def _store_in(tmp_path):
    return EnhanceDefaultsStore(
        user_path=str(tmp_path / "SPROUTS" / DEFAULTS_FILE_NAME),
        factory_path=str(tmp_path / "no_factory.json"),
    )


def test_load_never_raises_when_the_sweep_thread_cannot_start(tmp_path, monkeypatch, caplog):
    def refuse(self):
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr(threading.Thread, "start", refuse)
    store = _store_in(tmp_path)

    with caplog.at_level(logging.WARNING):
        settings = store.load()

    assert settings == enhancement.DEFAULT_SETTINGS
    assert store.sweep_thread is None
    assert "can't start new thread" in caplog.text


def test_a_later_load_sweeps_once_a_thread_can_start_again(tmp_path, monkeypatch):
    real_start = threading.Thread.start

    def refuse(self):
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr(threading.Thread, "start", refuse)
    _store_in(tmp_path).load()
    monkeypatch.setattr(threading.Thread, "start", real_start)

    store = _store_in(tmp_path)
    store.load()

    assert store.sweep_thread is not None  # the folder was not marked as swept
    store.sweep_thread.join(5)
