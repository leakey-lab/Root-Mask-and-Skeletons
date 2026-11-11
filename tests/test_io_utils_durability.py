"""``app.io_utils`` durable saves: the data is synced before the swap, the folder after it.

Promises: ``durable=True`` fsyncs the temp file before ``os.replace`` and, off Windows, the
folder after it; a folder that cannot be synced never fails a save; macOS syncs with
``F_BARRIERFSYNC`` (X1-05), falling back to the plain fsync; ``durable=False`` (bulk writes
of regenerable outputs) syncs nothing.
"""

from __future__ import annotations

import errno
import os
import stat
import sys

import cv2
import numpy as np
import pytest

from app import io_utils
from app.io_utils import (
    atomic_open,
    atomic_write_bytes,
    atomic_write_text,
    imwrite_unicode,
)


def _leftovers(directory) -> list[str]:
    return sorted(os.listdir(directory))


def _sample_image() -> np.ndarray:
    img = np.zeros((48, 64, 3), dtype=np.uint8)
    cv2.line(img, (2, 2), (60, 40), (255, 255, 255), 1)
    img[5:9, 5:9] = (10, 200, 30)
    return img


# ------------------------------------------------------------------ fsync and folder sync


@pytest.mark.parametrize("platform", ["win32", "linux", "darwin"])
def test_durable_writes_fsync_and_bulk_writes_can_skip_it(tmp_path, monkeypatch, platform):
    synced, folders = [], []
    monkeypatch.setattr(io_utils, "_fsync", synced.append)  # os.fsync, or macOS's barrier
    monkeypatch.setattr(io_utils, "_sync_folder", folders.append)
    monkeypatch.setattr(io_utils, "_current_platform", lambda: platform)
    synced_folders = [] if platform == "win32" else [os.path.abspath(tmp_path)]
    atomic_write_bytes(tmp_path / "hand_made.png", b"x")
    assert len(synced) == 1
    assert folders == synced_folders  # off Windows the rename is made durable too
    atomic_write_bytes(tmp_path / "generated.png", b"y", durable=False)
    atomic_write_text(tmp_path / "t.txt", "z", durable=False)
    with atomic_open(tmp_path / "s.csv", "w", durable=False) as fh:
        fh.write("a")
    imwrite_unicode(tmp_path / "g.png", _sample_image(), durable=False)
    assert len(synced) == 1
    assert folders == synced_folders
    assert (tmp_path / "generated.png").read_bytes() == b"y"
    assert sorted(os.listdir(tmp_path)) == [
        "g.png",
        "generated.png",
        "hand_made.png",
        "s.csv",
        "t.txt",
    ]


def test_the_folder_is_synced_after_the_swap_and_never_fails_a_save(tmp_path, monkeypatch):
    events = []
    real_replace = os.replace
    monkeypatch.setattr(io_utils, "_current_platform", lambda: "linux")
    monkeypatch.setattr(io_utils.os, "fsync", lambda fd: events.append("fsync"))
    monkeypatch.setattr(
        io_utils.os, "replace", lambda a, b: events.append("replace") or real_replace(a, b)
    )
    monkeypatch.setattr(io_utils, "_sync_folder", lambda folder: events.append(folder))
    atomic_write_bytes(tmp_path / "a.png", b"old")
    assert events == ["fsync", "replace", os.path.abspath(tmp_path)]

    def refuse(*_args):
        raise OSError(errno.EINVAL, "Invalid argument")

    monkeypatch.undo()
    monkeypatch.setattr(io_utils, "_current_platform", lambda: "linux")
    monkeypatch.setattr(io_utils.os, "open", refuse)  # the folder cannot be opened
    atomic_write_bytes(tmp_path / "a.png", b"new")
    assert (tmp_path / "a.png").read_bytes() == b"new"
    assert _leftovers(tmp_path) == ["a.png"]


def test_a_folder_sync_never_raises(tmp_path):
    io_utils._sync_folder(str(tmp_path))  # on Windows a folder cannot be opened: no error
    io_utils._sync_folder(str(tmp_path / "missing"))


@pytest.mark.skipif(sys.platform == "win32", reason="the folder fsync of Linux and macOS")
def test_posix_saves_fsync_the_file_then_swap_then_fsync_the_folder(tmp_path, monkeypatch):
    events = []
    # The real sync of this platform: os.fsync on Linux, F_BARRIERFSYNC on macOS (X1-05).
    real_fsync, real_replace = io_utils._fsync, os.replace

    def fsync(fd):
        events.append("folder" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file")
        real_fsync(fd)

    monkeypatch.setattr(io_utils, "_fsync", fsync)
    monkeypatch.setattr(os, "replace", lambda a, b: events.append("replace") or real_replace(a, b))
    atomic_write_bytes(tmp_path / "a.png", b"x")
    assert events == ["file", "replace", "folder"]

    def refuse_folders(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError(errno.EINVAL, "Invalid argument")  # a file system without it
        real_fsync(fd)

    monkeypatch.setattr(io_utils, "_fsync", refuse_folders)
    atomic_write_bytes(tmp_path / "a.png", b"new")
    assert (tmp_path / "a.png").read_bytes() == b"new"
    assert _leftovers(tmp_path) == ["a.png"]
    events.clear()
    monkeypatch.setattr(io_utils, "_fsync", fsync)
    atomic_write_bytes(tmp_path / "b.png", b"y", durable=False)
    assert events == ["replace"]  # a bulk write syncs nothing


# ------------------------------------------------------------------ macOS barrier sync (X1-05)


class _FakeFcntl:
    """Stands in for macOS's ``fcntl`` module; ``fail`` makes the call fail like SMB does."""

    def __init__(self, *, constant: int | None = None, fail: bool = False) -> None:
        self.calls: list[tuple[int, int]] = []
        self.fail = fail
        if constant is not None:
            self.F_BARRIERFSYNC = constant

    def fcntl(self, fd: int, cmd: int, arg: int = 0) -> int:
        self.calls.append((fd, cmd))
        if self.fail:
            raise OSError(errno.ENOTSUP, "Operation not supported")
        return 0


def _on(monkeypatch, platform: str, fake: _FakeFcntl | None) -> list[int]:
    """Pretend to run on ``platform`` with ``fake`` as fcntl; returns the ``os.fsync`` calls."""
    synced: list[int] = []
    monkeypatch.setattr(io_utils, "_current_platform", lambda: platform)
    monkeypatch.setattr(io_utils.os, "fsync", synced.append)
    # None in sys.modules makes "import fcntl" fail, as on Windows.
    monkeypatch.setitem(sys.modules, "fcntl", fake)
    return synced


def test_macos_durable_saves_sync_with_a_barrier(tmp_path, monkeypatch):
    """Python's fcntl has no F_BARRIERFSYNC (3.14), so <sys/fcntl.h>'s 85 is used."""
    fake = _FakeFcntl()
    synced = _on(monkeypatch, "darwin", fake)
    monkeypatch.setattr(io_utils, "_sync_folder", lambda folder: None)
    atomic_write_bytes(tmp_path / "a.png", b"x")
    assert [cmd for _fd, cmd in fake.calls] == [85]
    assert synced == []  # the barrier replaces the plain fsync
    assert (tmp_path / "a.png").read_bytes() == b"x"
    assert _leftovers(tmp_path) == ["a.png"]


def test_a_python_that_exports_the_barrier_constant_is_followed(tmp_path, monkeypatch):
    fake = _FakeFcntl(constant=1085)
    synced = _on(monkeypatch, "darwin", fake)
    monkeypatch.setattr(io_utils, "_sync_folder", lambda folder: None)
    atomic_write_bytes(tmp_path / "a.png", b"x")
    assert [cmd for _fd, cmd in fake.calls] == [1085] and synced == []


def test_macos_folder_sync_uses_the_barrier_too(tmp_path, monkeypatch):
    fake = _FakeFcntl()
    synced = _on(monkeypatch, "darwin", fake)
    closed = []
    monkeypatch.setattr(io_utils.os, "open", lambda path, flags: 99)
    monkeypatch.setattr(io_utils.os, "close", closed.append)
    io_utils._sync_folder(str(tmp_path))
    assert fake.calls == [(99, 85)] and synced == [] and closed == [99]


def test_a_file_system_without_the_barrier_gets_the_plain_fsync(tmp_path, monkeypatch):
    fake = _FakeFcntl(fail=True)
    synced = _on(monkeypatch, "darwin", fake)
    monkeypatch.setattr(io_utils, "_sync_folder", lambda folder: None)
    atomic_write_bytes(tmp_path / "a.png", b"x")
    assert len(fake.calls) == 1
    assert synced == [fake.calls[0][0]]  # the same file, synced the plain way
    assert (tmp_path / "a.png").read_bytes() == b"x"


def test_without_fcntl_a_mac_save_still_syncs(tmp_path, monkeypatch):
    synced = _on(monkeypatch, "darwin", None)
    monkeypatch.setattr(io_utils, "_sync_folder", lambda folder: None)
    atomic_write_bytes(tmp_path / "a.png", b"x")
    assert len(synced) == 1


@pytest.mark.parametrize("platform", ["win32", "linux"])
def test_windows_and_linux_keep_the_plain_fsync(tmp_path, monkeypatch, platform):
    fake = _FakeFcntl()
    synced = _on(monkeypatch, platform, fake)
    monkeypatch.setattr(io_utils, "_sync_folder", lambda folder: None)
    atomic_write_bytes(tmp_path / "a.png", b"x")
    assert fake.calls == [] and len(synced) == 1


def test_bulk_writes_sync_nothing_on_macos(tmp_path, monkeypatch):
    fake = _FakeFcntl()
    synced = _on(monkeypatch, "darwin", fake)
    atomic_write_bytes(tmp_path / "a.png", b"x", durable=False)
    assert fake.calls == [] and synced == []


@pytest.mark.skipif(sys.platform != "darwin", reason="the real F_BARRIERFSYNC of macOS")
def test_a_real_mac_save_issues_the_barrier(tmp_path, monkeypatch):
    import fcntl

    calls = []
    real_fcntl = fcntl.fcntl

    def spy(fd, cmd, *args):
        calls.append(cmd)
        return real_fcntl(fd, cmd, *args)

    monkeypatch.setattr(fcntl, "fcntl", spy)
    atomic_write_bytes(tmp_path / "a.png", b"x")
    assert calls and calls[0] == getattr(fcntl, "F_BARRIERFSYNC", 85)
    assert (tmp_path / "a.png").read_bytes() == b"x"


def test_a_failing_fsync_fails_the_save_and_keeps_the_old_file(tmp_path, monkeypatch):
    """A network drive or a full disk can refuse the sync: the save must fail (the data may
    not be on disk), keep the previous file and leave no temp file."""
    target = tmp_path / "gt_mask.png"
    target.write_bytes(b"old complete file")
    refused = OSError(errno.EIO, "Input/output error")

    def failing_fsync(fd):
        raise refused

    monkeypatch.setattr(io_utils, "_current_platform", lambda: "linux")  # the plain fsync
    monkeypatch.setattr(io_utils.os, "fsync", failing_fsync)
    with pytest.raises(OSError) as info:
        atomic_write_bytes(target, b"new")

    assert info.value is refused
    assert target.read_bytes() == b"old complete file"
    assert _leftovers(tmp_path) == ["gt_mask.png"]
