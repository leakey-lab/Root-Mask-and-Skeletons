"""``app.io_utils`` atomic writes: a save either replaces the whole file or leaves the old one.

Promises: data goes to a hidden ``.<name>.<8 hex>.tmp`` sibling that no scanner lists and that
fits wherever the target fits (MAX_PATH, 255 characters, 255 UTF-8 bytes); any failure keeps
the previous file and removes the temp file (R7: also when the close fails on a full disk);
a file held open by another program raises ``FileLockedError`` after short retries, a
read-only file does not; an error names the file being saved, never the temp file (C6-05);
``folder_not_writable`` tells a folder that takes no new files from one bad name (C6-01).
"""

from __future__ import annotations

import csv
import errno
import os
import stat
import sys
import threading
import time

import pytest

from app import io_utils
from app.io_utils import (
    FileLockedError,
    atomic_open,
    atomic_write_bytes,
    atomic_write_text,
)

UNICODE_DIRNAME = "Donn\u00e9es_\u6839\u7cfb"


def _leftovers(directory) -> list[str]:
    return sorted(os.listdir(directory))


def _sharing_violation(path, winerror: int = 32) -> PermissionError:
    exc = PermissionError(errno.EACCES, "The process cannot access the file", str(path))
    exc.winerror = winerror
    return exc


# ------------------------------------------------------------------ atomic writes


def test_atomic_write_bytes_creates_and_replaces(tmp_path):
    target = tmp_path / "a.bin"
    atomic_write_bytes(target, b"first")
    atomic_write_bytes(str(target), bytearray(b"second"))
    assert target.read_bytes() == b"second"
    assert _leftovers(tmp_path) == ["a.bin"]


def test_atomic_write_text_honours_encoding_and_newline(tmp_path):
    target = tmp_path / "t.csv"
    atomic_write_text(target, "a\nb\n", newline="")
    assert target.read_bytes() == b"a\nb\n"
    atomic_write_text(target, "\u00b5m\r\n", encoding="utf-8", newline="")
    assert target.read_bytes() == "\u00b5m\r\n".encode()


def test_atomic_open_streams_csv_identical_to_plain_open(tmp_path):
    rows = [{"Image": "x", "Area (mm\u00b2)": 1.25, "Error": ""}]
    headers = ["Image", "Area (mm\u00b2)", "Error"]

    def write(fh):
        writer = csv.DictWriter(fh, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    with open(tmp_path / "plain.csv", "w", newline="", encoding="utf-8") as fh:
        write(fh)
    with atomic_open(tmp_path / "atomic.csv", "w", newline="", encoding="utf-8") as fh:
        write(fh)
    assert (tmp_path / "atomic.csv").read_bytes() == (tmp_path / "plain.csv").read_bytes()


def test_failure_inside_atomic_open_keeps_old_file_and_no_temp(tmp_path):
    target = tmp_path / "keep.csv"
    target.write_bytes(b"old complete file")
    with pytest.raises(RuntimeError), atomic_open(target, "w") as fh:
        fh.write("partial")
        raise RuntimeError("killed mid-write")
    assert target.read_bytes() == b"old complete file"
    assert _leftovers(tmp_path) == ["keep.csv"]


def test_failed_replace_keeps_old_file_and_no_temp(tmp_path, monkeypatch):
    target = tmp_path / "m.png"
    target.write_bytes(b"old")

    def broken_replace(src, dst):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(io_utils.os, "replace", broken_replace)
    with pytest.raises(OSError) as info:
        atomic_write_bytes(target, b"new")
    assert not isinstance(info.value, FileLockedError)
    assert target.read_bytes() == b"old"
    assert _leftovers(tmp_path) == ["m.png"]


def test_temp_file_name_is_invisible_to_image_scanners(tmp_path, monkeypatch):
    seen = []
    real_replace = os.replace

    def spy(src, dst):
        seen.append(os.path.basename(src))
        real_replace(src, dst)

    monkeypatch.setattr(io_utils.os, "replace", spy)
    atomic_write_bytes(tmp_path / "x.png", b"png")
    (name,) = seen
    assert name.endswith(".tmp")
    assert not name.lower().endswith((".png", ".jpg", ".jpeg", ".csv", ".json"))


def test_missing_directory_raises_without_side_effects(tmp_path):
    target = tmp_path / "nope" / "x.bin"
    with pytest.raises(FileNotFoundError) as info:
        atomic_write_bytes(target, b"x")
    assert _leftovers(tmp_path) == []
    # (C6-05) The error names the file being saved, not the hidden temp file.
    assert info.value.filename == str(target)
    assert ".tmp" not in str(info.value)


def test_invalid_mode_is_rejected(tmp_path):
    with pytest.raises(ValueError), atomic_open(tmp_path / "x", "a"):
        pass


# ------------------------------------------------------------------ a failing close or flush (R7)


class _CloseFails:
    """A file whose first ``close()`` closes it, then raises like a flush on a full disk."""

    def __init__(self, fh):
        self._fh = fh
        self._failed = False

    def __getattr__(self, name):
        return getattr(self._fh, name)

    def close(self):
        self._fh.close()
        if not self._failed:
            self._failed = True
            raise OSError(errno.ENOSPC, "No space left on device (simulated)")


@pytest.fixture
def close_fails(monkeypatch):
    real_open = open
    monkeypatch.setattr(
        io_utils, "open", lambda *a, **k: _CloseFails(real_open(*a, **k)), raising=False
    )


def test_an_error_in_the_write_is_raised_and_the_temp_file_removed(tmp_path, close_fails):
    target = tmp_path / "root_lengths.csv"
    target.write_bytes(b"previous complete file")

    with pytest.raises(ValueError, match="bad row"), atomic_open(target, "w") as fh:
        fh.write("Image,Length (mm)\n")
        raise ValueError("bad row")

    assert os.listdir(tmp_path) == ["root_lengths.csv"]  # no temp file left behind
    assert target.read_bytes() == b"previous complete file"


def test_a_failing_flush_leaves_no_temp_file(tmp_path, monkeypatch, close_fails):
    def flush_fails(fh):
        raise OSError(errno.ENOSPC, "No space left on device (flush, simulated)")

    monkeypatch.setattr(_CloseFails, "flush", flush_fails, raising=False)

    with pytest.raises(OSError, match="flush, simulated"):
        atomic_write_bytes(tmp_path / "mask.png", b"\x89PNG" + b"\0" * 64)

    assert os.listdir(tmp_path) == []


# ------------------------------------------------------------------ locked files


def test_sharing_violation_retries_then_raises_file_locked(tmp_path, monkeypatch):
    target = tmp_path / "root_lengths.csv"
    target.write_text("old\n", encoding="utf-8")
    calls = []

    def locked_replace(src, dst):
        calls.append(dst)
        raise _sharing_violation(dst)

    monkeypatch.setattr(io_utils.os, "replace", locked_replace)
    monkeypatch.setattr(io_utils.time, "sleep", lambda s: None)
    with pytest.raises(FileLockedError) as info:
        atomic_write_text(target, "new\n")
    err = info.value
    assert isinstance(err, PermissionError)
    assert str(err) == (
        "root_lengths.csv is open in another program (for example Excel). Close it and try again."
    )
    assert err.filename == str(target)
    assert len(calls) == len(io_utils.REPLACE_RETRY_DELAYS) + 1
    assert target.read_text(encoding="utf-8") == "old\n"
    assert _leftovers(tmp_path) == ["root_lengths.csv"]


def test_transient_sharing_violation_is_retried(tmp_path, monkeypatch):
    target = tmp_path / "x.png"
    real_replace = os.replace
    attempts = []

    def flaky_replace(src, dst):
        attempts.append(dst)
        if len(attempts) == 1:
            raise _sharing_violation(dst, winerror=5)
        real_replace(src, dst)

    monkeypatch.setattr(io_utils.os, "replace", flaky_replace)
    monkeypatch.setattr(io_utils.time, "sleep", lambda s: None)
    atomic_write_bytes(target, b"ok")
    assert target.read_bytes() == b"ok"
    assert len(attempts) == 2


def test_plain_permission_error_is_not_reported_as_locked(tmp_path, monkeypatch):
    def denied(src, dst):
        raise PermissionError(errno.EACCES, "Permission denied", dst)  # no winerror

    monkeypatch.setattr(io_utils.os, "replace", denied)
    with pytest.raises(PermissionError) as info:
        atomic_write_bytes(tmp_path / "x.bin", b"x")
    assert not isinstance(info.value, FileLockedError)


def test_file_locked_error_survives_pickling():
    import pickle

    err = pickle.loads(pickle.dumps(FileLockedError("D:/data/root_areas.csv")))
    assert isinstance(err, FileLockedError)
    assert "root_areas.csv is open in another program" in str(err)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows share-mode lock")
def test_real_excel_style_lock_raises_file_locked(tmp_path):
    import ctypes
    from ctypes import wintypes

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateFileW.restype = wintypes.HANDLE
    k32.CreateFileW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    target = tmp_path / "root_lengths.csv"
    target.write_text("old\n", encoding="utf-8")
    generic_read, file_share_read, open_existing = 0x80000000, 0x1, 3
    handle = k32.CreateFileW(
        str(target), generic_read, file_share_read, None, open_existing, 0, None
    )
    try:
        with pytest.raises(FileLockedError):
            atomic_write_text(target, "new\n")
    finally:
        k32.CloseHandle(handle)
    assert target.read_text(encoding="utf-8") == "old\n"
    assert _leftovers(tmp_path) == ["root_lengths.csv"]


@pytest.mark.skipif(sys.platform != "win32", reason="Windows read-only attribute")
def test_read_only_target_is_not_reported_as_locked(tmp_path, monkeypatch):
    import stat

    target = tmp_path / "ro.csv"
    target.write_text("old\n", encoding="utf-8")
    os.chmod(target, stat.S_IREAD)
    monkeypatch.setattr(io_utils.time, "sleep", lambda s: pytest.fail("must not retry"))
    try:
        with pytest.raises(PermissionError) as info:
            atomic_write_text(target, "new\n")
        assert not isinstance(info.value, FileLockedError)
        assert target.read_text(encoding="utf-8") == "old\n"
    finally:
        os.chmod(target, stat.S_IWRITE | stat.S_IREAD)
    assert _leftovers(tmp_path) == ["ro.csv"]
    # (C6-05) "[WinError 5] Access is denied: 'ro.csv'", not "'.ro.csv.<hex>.tmp' -> 'ro.csv'".
    assert info.value.filename == str(target)
    assert ".tmp" not in str(info.value) and " -> " not in str(info.value)
    # A read-only file is that file's problem, not the folder's (C6-01).
    assert io_utils.folder_not_writable(info.value) is None


# ------------------------------------------------------------------ temp names that fit (O21, X1)


def test_temp_name_fits_255_utf8_bytes_on_posix(tmp_path):
    """ext4 and APFS cap a name at 255 UTF-8 bytes, not characters: 80 CJK characters plus
    ".png" is 244 bytes, its temp name 14 characters more (O21)."""
    target = tmp_path / ("\u6839" * 80 + ".png")
    name = os.path.basename(io_utils._temp_path(str(target), "linux"))
    assert len(name.encode("utf-8")) <= 255
    assert io_utils._is_temp_name(name)
    assert name.startswith(".\u6839")  # the name is cut from its end, not emptied


def test_temp_name_keeps_all_characters_on_windows(tmp_path):
    """NTFS counts UTF-16 units, so the whole 84-character name stays."""
    target = tmp_path / ("\u6839" * 80 + ".png")
    name = os.path.basename(io_utils._temp_path(str(target), "win32"))
    assert name.startswith("." + "\u6839" * 80 + ".png.")


def test_a_cjk_name_near_the_byte_limit_is_written_atomically(tmp_path):
    target = tmp_path / ("\u6839" * 80 + ".png")
    atomic_write_bytes(target, b"data")
    assert target.read_bytes() == b"data"
    assert _leftovers(tmp_path) == [target.name]


def test_temp_name_of_a_non_utf8_file_name_fits_255_bytes_on_posix(tmp_path):
    """os.listdir returns an undecodable name with surrogate escapes; counting its bytes
    must not raise, or no atomic write to such a file would work."""
    target = tmp_path / ("a\udcff" * 120 + ".png")
    name = os.path.basename(io_utils._temp_path(str(target), "linux"))
    assert len(os.fsencode(name)) <= 255
    assert io_utils._is_temp_name(name)


def test_a_surrogate_escaped_name_is_written_atomically(tmp_path):
    target = tmp_path / "a\udcff.png"
    try:
        atomic_write_bytes(target, b"data")
    except OSError, UnicodeError:
        pytest.skip("this file system cannot hold a non-UTF-8 name")
    assert target.read_bytes() == b"data"
    assert _leftovers(tmp_path) == [target.name]


_MAX_PATH = 259  # Windows MAX_PATH less the terminating NUL


_NAME = "24EFCAM1_T001_L011_2024.07.18_111618_001_SMM_fake.png"


def _padded_folder(base: str, length: int) -> str:
    """A folder path (not created) whose absolute form is exactly ``length`` characters."""
    base = os.path.abspath(base)
    need = length - len(base) - 1
    assert need > 0
    parts, left = [], need
    while left > 0:
        chunk = min(left, 40)
        if left - chunk == 1:  # never leave room for a lone separator
            chunk -= 1
        parts.append("d" * chunk)
        left -= chunk + 1
    folder = os.path.join(base, *parts)
    assert len(folder) == length, (len(folder), length)
    return folder


def test_a_file_name_near_the_255_character_limit_is_written_atomically(tmp_path):
    target = tmp_path / ("a" * 246 + ".png")  # 250 characters: a valid file name
    with open(target, "wb") as fh:  # a plain write works...
        fh.write(b"plain")

    io_utils.atomic_write_bytes(target, b"atomic")  # ...and so must the atomic one

    assert target.read_bytes() == b"atomic"
    assert os.listdir(tmp_path) == [target.name]  # no temp file left behind


@pytest.mark.parametrize("length", [200, 240, 245, 246, 250, 255, 259])
def test_the_temp_path_fits_wherever_the_target_path_fits(tmp_path, length):
    folder = _padded_folder(str(tmp_path), length - len(_NAME) - 1)
    target = os.path.join(folder, _NAME)
    assert len(target) == length

    temp = io_utils._temp_path(target)

    assert len(os.path.abspath(temp)) <= _MAX_PATH
    assert os.path.dirname(temp) == folder  # a sibling: os.replace stays on one volume
    name = os.path.basename(temp)
    assert name.startswith(".") and name.endswith(".tmp")  # hidden from every scanner
    assert io_utils._is_temp_name(name)  # the stale-temp sweep still finds it


def test_a_short_path_keeps_the_whole_target_name_in_the_temp_name(tmp_path):
    target = os.path.join(str(tmp_path), _NAME)

    name = os.path.basename(io_utils._temp_path(target))

    head, token, suffix = name.rsplit(".", 2)
    assert head == "." + _NAME and suffix == "tmp" and len(token) == 8


def test_a_path_longer_than_max_path_keeps_the_whole_target_name(tmp_path):
    # Such a target exists only where long paths are enabled: nothing to shorten for.
    folder = _padded_folder(str(tmp_path), 300)
    name = os.path.basename(io_utils._temp_path(os.path.join(folder, _NAME)))
    assert name.startswith("." + _NAME + ".")


def test_a_stale_shortened_temp_file_is_swept(tmp_path):
    target = tmp_path / ("b" * 246 + ".png")
    temp = io_utils._temp_path(str(target))
    with open(temp, "wb") as fh:
        fh.write(b"left by a killed save")
    old = time.time() - 2 * 3600
    os.utime(temp, (old, old))

    removed = io_utils.sweep_stale_temp_files(tmp_path)

    assert removed == [temp] and not os.path.exists(temp)


def test_a_near_max_path_target_is_written_atomically(tmp_path):
    folder = _padded_folder(str(tmp_path), _MAX_PATH - len(_NAME) - 1)
    try:
        os.makedirs(folder)
    except OSError as exc:  # pragma: no cover - the folder itself is within MAX_PATH
        pytest.skip(f"cannot create the test folder: {exc}")
    target = os.path.join(folder, _NAME)
    assert len(target) == _MAX_PATH

    io_utils.atomic_write_bytes(target, b"png bytes")

    with open(target, "rb") as fh:
        assert fh.read() == b"png bytes"
    assert os.listdir(folder) == [_NAME]


# ------------------------------------------------------------------ errors name the target (C6-05)


def _refuse_new_files(monkeypatch, err: int = errno.EACCES, strerror: str = "Permission denied"):
    """The folder refuses new files: creating the temp file fails as the OS would say it."""
    real_open = open

    def refusing_open(path, mode="r", *args, **kwargs):
        if "x" in mode:
            raise OSError(err, strerror, path)
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(io_utils, "open", refusing_open, raising=False)


def test_a_temp_file_that_cannot_be_made_names_the_file_being_saved(tmp_path, monkeypatch):
    target = tmp_path / "root_lengths.csv"
    _refuse_new_files(monkeypatch)
    with pytest.raises(PermissionError) as info:
        atomic_write_text(target, "new\n")
    assert info.value.filename == str(target)
    assert str(info.value) == f"[Errno {errno.EACCES}] Permission denied: {str(target)!r}"
    assert _leftovers(tmp_path) == []


def test_a_failed_swap_names_the_file_being_saved_not_the_temp_file(tmp_path, monkeypatch):
    target = tmp_path / "root_areas.csv"
    target.write_text("old\n", encoding="utf-8")

    def broken_replace(src, dst):
        raise OSError(errno.EIO, "Input/output error", src, None, dst)  # names both

    monkeypatch.setattr(io_utils.os, "replace", broken_replace)
    with pytest.raises(OSError) as info:
        atomic_write_text(target, "new\n")
    err = info.value
    assert err.errno == errno.EIO
    assert err.filename == str(target) and err.filename2 is None
    assert ".tmp" not in str(err) and " -> " not in str(err)
    assert repr(str(target)) in str(err)
    assert target.read_text(encoding="utf-8") == "old\n"
    assert _leftovers(tmp_path) == ["root_areas.csv"]


def test_errors_that_name_other_files_are_left_alone(tmp_path):
    other = str(tmp_path / "elsewhere.txt")
    with pytest.raises(OSError) as info, atomic_open(tmp_path / "a.csv") as fh:
        fh.write("x")
        raise FileNotFoundError(errno.ENOENT, "No such file or directory", other)
    assert info.value.filename == other
    assert _leftovers(tmp_path) == []


# ------------------------------------------------------------------ unwritable folders (C6-01)


@pytest.mark.parametrize(
    ("err", "strerror"),
    [
        (errno.EACCES, "Permission denied"),
        (errno.EPERM, "Operation not permitted"),
        (errno.EROFS, "Read-only file system"),
        (errno.ENOENT, "No such file or directory"),
        (errno.ENOTDIR, "Not a directory"),
    ],
)
def test_a_folder_that_takes_no_new_files_is_named(tmp_path, monkeypatch, err, strerror):
    folder = tmp_path / "out"
    if err not in (errno.ENOENT, errno.ENOTDIR):
        folder.mkdir()  # "no such file" is the folder's only when the folder is gone
    _refuse_new_files(monkeypatch, err, strerror)
    with pytest.raises(OSError) as info:
        atomic_write_bytes(folder / "a_fake.png", b"png")
    assert info.value.errno == err  # still the OS's own error, for every other caller

    folder_error = io_utils.folder_not_writable(info.value)
    assert isinstance(folder_error, io_utils.FolderNotWritableError)
    assert isinstance(folder_error, OSError)
    assert folder_error.folder == os.path.abspath(folder)
    assert folder_error.errno == err
    assert folder_error.__cause__ is info.value
    assert str(folder_error) == (
        f"Cannot save files in {os.path.abspath(folder)}: {strerror}. "
        "Check that the folder exists and that you may write to it, then try again."
    )


@pytest.mark.parametrize(
    ("err", "strerror"),
    [(errno.ENOENT, "No such file or directory"), (errno.ENOTDIR, "Not a directory")],
)
def test_no_such_file_in_a_folder_that_exists_is_about_the_one_name(
    tmp_path, monkeypatch, err, strerror
):
    """Windows says ENOENT for a path over MAX_PATH (long paths off, the default): a
    generated name longer than its input can give one while the folder is there. That is
    one image's failure; the run must not stop over it."""
    _refuse_new_files(monkeypatch, err, strerror)
    with pytest.raises(OSError) as info:
        atomic_write_bytes(tmp_path / "a_fake.png", b"png")
    assert info.value.errno == err
    assert info.value.filename == str(tmp_path / "a_fake.png")  # (C6-05)
    assert io_utils.folder_not_writable(info.value) is None
    assert _leftovers(tmp_path) == []


@pytest.mark.parametrize(
    "err",
    [errno.EINVAL, errno.ENAMETOOLONG, errno.EEXIST, errno.EMFILE, errno.EIO],
)
def test_temp_file_errors_of_one_name_or_the_moment_are_not_the_folders(tmp_path, monkeypatch, err):
    _refuse_new_files(monkeypatch, err, os.strerror(err))
    with pytest.raises(OSError) as info:
        atomic_write_bytes(tmp_path / "a_fake.png", b"png")
    assert io_utils.folder_not_writable(info.value) is None


def test_only_the_temp_file_step_tells_of_the_folder(tmp_path, monkeypatch):
    """A swap or a write that fails is about the one file; a bare error is about nothing."""

    def denied(src, dst):
        raise PermissionError(errno.EACCES, "Permission denied", dst)

    monkeypatch.setattr(io_utils.os, "replace", denied)
    with pytest.raises(PermissionError) as info:
        atomic_write_bytes(tmp_path / "x.png", b"x")
    assert io_utils.folder_not_writable(info.value) is None
    assert io_utils.folder_not_writable(PermissionError(errno.EACCES, "denied", "x")) is None
    assert io_utils.folder_not_writable(ValueError("not an OSError")) is None


def test_a_missing_folder_is_named_for_real(tmp_path):
    folder = tmp_path / "mask"
    with pytest.raises(FileNotFoundError) as info:
        atomic_write_bytes(folder / "a.png", b"png")
    folder_error = io_utils.folder_not_writable(info.value)
    assert folder_error is not None and folder_error.folder == str(folder)
    assert str(folder) in str(folder_error) and ".tmp" not in str(folder_error)


def test_a_folder_that_is_a_file_is_named_for_real(tmp_path):
    """The output "folder" is a file (ENOTDIR on POSIX, "path not found" on Windows)."""
    folder = tmp_path / "mask"
    folder.write_bytes(b"not a folder")
    with pytest.raises(OSError) as info:
        atomic_write_bytes(folder / "a.png", b"png")
    folder_error = io_utils.folder_not_writable(info.value)
    assert folder_error is not None and folder_error.folder == str(folder)
    assert folder.read_bytes() == b"not a folder"


@pytest.mark.skipif(
    sys.platform == "win32" or getattr(os, "geteuid", lambda: 0)() == 0,
    reason="POSIX folder permissions (root ignores them)",
)
def test_a_folder_without_write_permission_is_named_for_real(tmp_path):
    folder = tmp_path / "protected"
    folder.mkdir()
    os.chmod(folder, stat.S_IRUSR | stat.S_IXUSR)
    try:
        with pytest.raises(PermissionError) as info:
            atomic_write_bytes(folder / "a.png", b"png")
    finally:
        os.chmod(folder, stat.S_IRWXU)
    assert info.value.filename == str(folder / "a.png")
    folder_error = io_utils.folder_not_writable(info.value)
    assert folder_error is not None and folder_error.folder == str(folder)
    assert _leftovers(folder) == []


def test_folder_not_writable_error_survives_pickling():
    import pickle

    err = io_utils.FolderNotWritableError("D:/data/mask", "Permission denied", errno.EACCES)
    back = pickle.loads(pickle.dumps(err))
    assert isinstance(back, io_utils.FolderNotWritableError)
    assert (back.folder, back.errno, str(back)) == (err.folder, err.errno, str(err))


# ------------------------------------------------------------------ interrupts, readers, racing writers


@pytest.mark.parametrize("where", ["block", "swap"])
def test_an_interrupt_keeps_the_old_file_removes_the_temp_file_and_propagates(
    tmp_path, monkeypatch, where
):
    """Ctrl+C in the middle of a save, or a worker's SystemExit at the swap: the previous file
    stays, no hidden temp file (and no open handle) is left, the interrupt itself goes on."""
    target = tmp_path / "root_lengths.csv"
    target.write_bytes(b"old complete file")
    interrupt = KeyboardInterrupt() if where == "block" else SystemExit(1)
    if where == "swap":

        def interrupted_replace(src, dst):
            raise interrupt

        monkeypatch.setattr(io_utils.os, "replace", interrupted_replace)

    with pytest.raises(type(interrupt)) as info, atomic_open(target, "w") as fh:
        fh.write("partial")
        if where == "block":
            raise interrupt

    assert info.value is interrupt
    assert target.read_bytes() == b"old complete file"
    assert _leftovers(tmp_path) == ["root_lengths.csv"]


def test_while_a_save_runs_the_target_holds_the_previous_complete_file(tmp_path):
    target = tmp_path / "root_areas.csv"
    target.write_bytes(b"previous complete file")

    with atomic_open(target, "wb") as fh:
        fh.write(b"new data, not yet swapped in")
        fh.flush()
        seen_during = target.read_bytes()  # what a reader (the metrics bar) opens now

    assert seen_during == b"previous complete file"
    assert target.read_bytes() == b"new data, not yet swapped in"


def test_a_new_target_does_not_exist_until_its_save_ends(tmp_path):
    target = tmp_path / "mask.png"

    with atomic_open(target, "wb") as fh:
        fh.write(b"png")
        exists_during = target.exists()

    assert not exists_during
    assert target.read_bytes() == b"png"


def test_two_writers_of_one_file_leave_one_whole_file(tmp_path):
    """Two threads save one file 50 times each (an editor save racing a hand-back): the file
    is always one writer's complete data, never a mix, and no temp file is left. A save may
    fail only as a lock (the other writer's swap on Windows), never otherwise. (What a reader
    sees during a save: ``test_while_a_save_runs_the_target_holds_the_previous_complete_file``.)"""
    target = tmp_path / "mask.png"
    payloads = (b"a" * 200_000, b"b" * 200_000)
    failures: list[BaseException] = []

    def write(data: bytes) -> None:
        for _ in range(50):
            try:
                atomic_write_bytes(target, data, durable=False)
            except FileLockedError:
                pass  # the promised answer to a file another writer holds
            except BaseException as exc:  # noqa: BLE001 - recorded, asserted below
                failures.append(exc)

    writers = [threading.Thread(target=write, args=(data,)) for data in payloads]
    for thread in writers:
        thread.start()
    for thread in writers:
        thread.join()

    assert failures == []
    assert target.read_bytes() in payloads
    assert _leftovers(tmp_path) == ["mask.png"]
