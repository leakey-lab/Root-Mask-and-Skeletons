"""Foundation-core: atomic writes, locked-file errors and Unicode-safe image I/O."""

from __future__ import annotations

import csv
import errno
import os
import stat
import sys

import cv2
import numpy as np
import pytest

from app import io_utils
from app.io_utils import (
    FileLockedError,
    atomic_open,
    atomic_write_bytes,
    atomic_write_text,
    imread_unicode,
    imwrite_unicode,
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


# ------------------------------------------------------------------ image I/O


def _sample_image() -> np.ndarray:
    img = np.zeros((48, 64, 3), dtype=np.uint8)
    cv2.line(img, (2, 2), (60, 40), (255, 255, 255), 1)
    img[5:9, 5:9] = (10, 200, 30)
    return img


def test_imwrite_then_imread_round_trips_on_unicode_path(tmp_path):
    folder = tmp_path / UNICODE_DIRNAME
    folder.mkdir()
    path = folder / "24EFCAM1_T001_L045_2024.07.18_111618_001_SMM_fake.png"
    img = _sample_image()
    imwrite_unicode(path, img)
    back = imread_unicode(path)
    assert back is not None and np.array_equal(back, img)
    gray = imread_unicode(str(path), cv2.IMREAD_GRAYSCALE)
    assert gray.shape == img.shape[:2]


def test_imread_unicode_matches_cv2_imread_on_ascii_path(tmp_path):
    path = tmp_path / "plain.png"
    cv2.imwrite(str(path), _sample_image())
    for flags in (cv2.IMREAD_COLOR, cv2.IMREAD_GRAYSCALE, cv2.IMREAD_UNCHANGED):
        assert np.array_equal(imread_unicode(path, flags), cv2.imread(str(path), flags))


def test_imread_unicode_returns_none_for_missing_empty_or_corrupt(tmp_path):
    assert imread_unicode(tmp_path / "missing.png") is None
    (tmp_path / "empty.png").write_bytes(b"")
    assert imread_unicode(tmp_path / "empty.png") is None
    (tmp_path / "corrupt.png").write_bytes(b"not a png at all")
    assert imread_unicode(tmp_path / "corrupt.png") is None
    assert imread_unicode(tmp_path) is None  # a directory


def test_imwrite_unicode_raises_on_encode_failure_and_leaves_no_file(tmp_path):
    with pytest.raises(ValueError):
        imwrite_unicode(tmp_path / "bad.unknownext", _sample_image())
    assert _leftovers(tmp_path) == []


def test_imwrite_unicode_passes_encoder_params(tmp_path):
    img = _sample_image()
    imwrite_unicode(tmp_path / "fast.png", img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    imwrite_unicode(tmp_path / "small.png", img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    assert (tmp_path / "fast.png").stat().st_size > (tmp_path / "small.png").stat().st_size
    assert np.array_equal(imread_unicode(tmp_path / "fast.png"), img)


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
