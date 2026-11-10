"""``MaskStore.read`` of model masks (8-bit grey PNG) and of broken PNGs.

A model mask takes a Pillow fast path, like an editor mask (8-bit RGBA, R == G == B).
Both must read exactly as the old read did: ``imread_unicode(path, IMREAD_GRAYSCALE)``
thresholded at ``MASK_THRESHOLD``, ``MaskFileError`` where that returned None. The file
is now read once and decoded from its bytes.

Pillow and libpng (the reference decode) disagree on some broken files: Pillow accepts a
zlib stream that ends early or never ends, and ignores parts of the IHDR. The fast path
must leave those to the reference decode, so they are tested here for both kinds.
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np
import pytest
from PIL import Image

import app.masks.store as store_module
from app.io_utils import imread_unicode
from app.masks import MASK_THRESHOLD, MaskFileError, MaskStore
from tests.mask_corpus_support import ALL, GREY, _pil_png


def _old_read(store: MaskStore, stem: str) -> np.ndarray | None:
    """``MaskStore.read`` before any fast path, copied as the reference."""
    path = store.find(stem)
    if path is None:
        return None
    gray = imread_unicode(path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise MaskFileError(f"cannot decode {path}")
    return gray > MASK_THRESHOLD


def _outcome(read, store: MaskStore, stem: str):
    """The array a read returns, or the text of the MaskFileError it raises."""
    try:
        return read(store, stem)
    except MaskFileError as exc:
        return f"MaskFileError: {exc}"


def _same(a, b) -> bool:
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b)
    return type(a) is type(b) and a == b


def _write(tmp_path, data: bytes) -> MaskStore:
    store = MaskStore.for_image(str(tmp_path / "a.png"))
    os.makedirs(store.directory, exist_ok=True)
    with open(store.path_for("a"), "wb") as fh:
        fh.write(data)
    return store


@pytest.mark.parametrize("name", sorted(ALL))
def test_read_is_the_old_read(tmp_path, name):
    store = _write(tmp_path, ALL[name][0])
    old = _outcome(_old_read, store, "a")
    assert _same(_outcome(MaskStore.read, store, "a"), old)
    assert _same(_outcome(MaskStore.read, store, "a"), old)  # the same again


@pytest.mark.parametrize("name", sorted(ALL))
def test_only_the_expected_files_take_the_fast_path(name):
    data, fast = ALL[name]
    plane = store_module._decode_plain_png(data)
    assert (plane is not None) == fast
    if plane is not None:  # the plane itself, before the threshold, is the reference one
        reference = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
        assert plane.dtype == np.uint8 and np.array_equal(plane, reference)


@pytest.mark.parametrize("name", sorted(ALL))
def test_the_bytes_decode_like_the_file_decode(tmp_path, name):
    store = _write(tmp_path, ALL[name][0])
    from_file = imread_unicode(store.path_for("a"), cv2.IMREAD_GRAYSCALE)
    from_bytes = store_module._decode_reference(ALL[name][0])
    assert (from_file is None) == (from_bytes is None)
    if from_file is not None:
        assert np.array_equal(from_file, from_bytes)


def test_random_model_masks_read_back_exactly(tmp_path):
    rng = np.random.default_rng(17)
    for _ in range(40):
        h, w = (int(v) for v in rng.integers(1, 300, 2))
        grey = (rng.random((h, w)) < 0.3).astype(np.uint8) * 255
        grey[rng.random((h, w)) < 0.05] = rng.integers(190, 211)
        data = _pil_png(Image.fromarray(grey, "L"), compress_level=int(rng.integers(0, 10)))
        store = _write(tmp_path, data)
        assert store_module._decode_plain_png(data) is not None
        assert np.array_equal(store.read("a"), grey > MASK_THRESHOLD)
        assert np.array_equal(store.read("a"), _old_read(store, "a"))


# ------------------------------------------------------------------- files that cannot be read
def _missing(tmp_path, monkeypatch) -> MaskStore:
    """A store whose mask vanishes between find() and the read."""
    store = _write(tmp_path, GREY["filter_none"][0])
    path = store.path_for("a")
    os.remove(path)
    monkeypatch.setattr(MaskStore, "find", lambda self, stem: path)
    return store


def test_a_vanished_file_raises_as_before(tmp_path, monkeypatch):
    store = _missing(tmp_path, monkeypatch)
    old = _outcome(_old_read, store, "a")
    assert old.startswith("MaskFileError")
    assert _outcome(MaskStore.read, store, "a") == old


def test_a_folder_in_place_of_the_file_raises_as_before(tmp_path, monkeypatch):
    store = MaskStore.for_image(str(tmp_path / "a.png"))
    os.makedirs(store.path_for("a"))
    monkeypatch.setattr(MaskStore, "find", lambda self, stem: self.path_for(stem))
    old = _outcome(_old_read, store, "a")
    assert old.startswith("MaskFileError")
    assert _outcome(MaskStore.read, store, "a") == old


@pytest.mark.skipif(sys.platform != "win32", reason="an exclusive lock is a Windows thing")
def test_a_locked_file_raises_as_before(tmp_path):
    import ctypes
    from ctypes import wintypes

    store = _write(tmp_path, GREY["filter_none"][0])
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileW.restype = wintypes.HANDLE
    kernel32.CreateFileW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    generic_read, no_sharing, open_existing = 0x80000000, 0, 3
    handle = kernel32.CreateFileW(
        store.path_for("a"), generic_read, no_sharing, None, open_existing, 0, None
    )
    assert handle not in (None, wintypes.HANDLE(-1).value)
    try:
        old = _outcome(_old_read, store, "a")
        assert old.startswith("MaskFileError")
        assert _outcome(MaskStore.read, store, "a") == old
    finally:
        kernel32.CloseHandle(handle)
    assert isinstance(store.read("a"), np.ndarray)  # readable again once released
