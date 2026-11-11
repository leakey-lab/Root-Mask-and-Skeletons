"""The mask files of ``app.masks``: the reader of ``mask/`` and the GT mask writer.

The saved mask decides the measured root area. Promises pinned here: paths follow the project
layout; a saved mask is white on transparent (RGBA 0/255, the bytes the Qt writer of before
made, far smaller), deterministic and written atomically (a failed save keeps the old file and
leaves no temp); what is written reads back identical, also at a path with spaces and
non-ASCII letters; a read is the grey decode above 127 (owner decision O10: alpha ignored,
colours through the luma weights, a legacy JPEG too); a file that exists but cannot be read
raises ``MaskFileError`` and is never taken for an empty mask; and the area Measure writes is
exactly the saved pixels times the area of a pixel. Every file kind against the reference
decode: ``tests/test_mask_store_decode.py``.
"""

import os
import sys

import cv2
import numpy as np
import pytest

from app.config import Calibration
from app.inference import metrics
from app.masks import (
    MASK_THRESHOLD,
    MaskFileError,
    MaskStore,
    encode_mask_png,
    read_mask_file,
    write_mask_file,
)
from app.project_layout import ProjectLayout
from tests.correct_support import box_plane, write_mask
from tests.mask_corpus_support import _mask


def _qt_reference_png(path, on):
    """What the editor wrote before (QImage ARGB32, quality 100)."""
    from PyQt6.QtGui import QImage

    out = np.zeros((*on.shape, 4), np.uint8)
    out[on] = 255
    h, w = on.shape
    assert QImage(out.data, w, h, w * 4, QImage.Format.Format_ARGB32).save(str(path), "PNG", 100)


def test_paths_follow_the_project_layout(tmp_path):
    store = MaskStore.for_image(str(tmp_path / "plot_001.jpg"))
    assert store.path_for("plot_001") == os.path.normpath(str(tmp_path / "mask" / "plot_001.png"))
    assert store.find("plot_001") is None
    assert store.read("plot_001") is None


def test_write_is_pixel_identical_to_the_qt_writer_and_far_smaller(tmp_path, app):
    on = np.zeros((480, 640), bool)
    on[100:110, 50:600] = True
    store = MaskStore.for_image(str(tmp_path / "a.jpg"))
    path = store.path_for("a")
    write_mask_file(path, on)
    ref = tmp_path / "ref.png"
    _qt_reference_png(ref, on)

    new = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    old = cv2.imread(str(ref), cv2.IMREAD_UNCHANGED)
    assert new.shape == old.shape == (480, 640, 4)
    assert np.array_equal(new, old)  # white on transparent, alpha only 0/255
    assert os.path.getsize(path) < 64 * 1024 < os.path.getsize(ref)
    assert [p for p in os.listdir(tmp_path / "mask") if p != "a.png"] == []  # no temp left


def test_round_trip_and_grayscale_readers_see_the_same_mask(tmp_path):
    on = _mask()
    store = MaskStore.for_image(str(tmp_path / "plot_001.png"))
    write_mask_file(store.path_for("plot_001"), on)
    assert np.array_equal(store.read("plot_001"), on)
    gray = cv2.imread(store.path_for("plot_001"), cv2.IMREAD_GRAYSCALE)
    assert set(np.unique(gray)) == {0, 255}
    assert np.array_equal(gray > MASK_THRESHOLD, on)


def test_a_mask_saved_under_a_path_with_spaces_and_accents_reads_back(tmp_path):
    folder = tmp_path / "Wurzeln Feld ä" / "Été 2024"
    store = MaskStore.for_image(str(folder / "T001_L044 ö.jpg"))
    path = os.path.join(str(folder), "gt", "mask", "T001_L044 ö.png")
    on = _mask()

    write_mask_file(path, on)  # the folders are created

    assert os.path.isfile(path)
    assert np.array_equal(read_mask_file(path), on)
    write_mask_file(store.path_for("T001_L044 ö"), on)
    assert np.array_equal(store.read("T001_L044 ö"), on)


def test_png_bytes_are_deterministic():
    assert encode_mask_png(_mask()) == encode_mask_png(_mask())


def test_failed_replace_keeps_the_previous_file(tmp_path, monkeypatch):
    store = MaskStore.for_image(str(tmp_path / "a.png"))
    path = store.path_for("a")
    write_mask_file(path, _mask())
    before = open(path, "rb").read()

    def _boom(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", _boom)
    with pytest.raises(OSError):
        write_mask_file(path, np.zeros((60, 80), bool) | True)
    assert open(path, "rb").read() == before
    assert sorted(os.listdir(tmp_path / "mask")) == ["a.png"]


# ------------------------------------------------------------------ files that cannot be read
def test_unreadable_file_raises(tmp_path):
    store = MaskStore.for_image(str(tmp_path / "a.png"))
    os.makedirs(store.directory)
    with open(store.path_for("a"), "wb") as fh:
        fh.write(b"not a png")
    with pytest.raises(MaskFileError):
        store.read("a")


def test_a_file_found_by_the_store_that_vanished_before_the_read_raises(tmp_path, monkeypatch):
    store = MaskStore.for_image(str(tmp_path / "a.png"))
    path = store.path_for("a")  # never written: the lookup found it, then it went
    monkeypatch.setattr(MaskStore, "find", lambda self, stem: path)
    with pytest.raises(MaskFileError):
        store.read("a")


def test_a_folder_in_place_of_the_file_raises(tmp_path, monkeypatch):
    store = MaskStore.for_image(str(tmp_path / "a.png"))
    os.makedirs(store.path_for("a"))
    monkeypatch.setattr(MaskStore, "find", lambda self, stem: self.path_for(stem))
    with pytest.raises(MaskFileError):
        store.read("a")


@pytest.mark.skipif(sys.platform != "win32", reason="an exclusive lock is a Windows thing")
def test_a_locked_file_raises_and_reads_once_released(tmp_path):
    import ctypes
    from ctypes import wintypes

    store = MaskStore.for_image(str(tmp_path / "a.png"))
    write_mask_file(store.path_for("a"), _mask())
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
        with pytest.raises(MaskFileError):
            store.read("a")
    finally:
        kernel32.CloseHandle(handle)
    assert np.array_equal(store.read("a"), _mask())  # readable again once released


def test_the_store_reads_the_file_found_without_a_second_lookup(tmp_path, monkeypatch):
    found = write_mask(os.path.join(str(tmp_path), "mask", "img.png"), box_plane())
    store = MaskStore(ProjectLayout(str(tmp_path)))
    monkeypatch.setattr(ProjectLayout, "find_mask", lambda *_args: pytest.fail("looked up"))

    assert np.array_equal(store.read("img", path=found), box_plane() > 0)
    os.remove(found)
    assert store.read("img", path=found) is None  # gone since the lookup: no prediction
    with open(found, "wb") as fh:
        fh.write(b"not a png")
    with pytest.raises(MaskFileError):
        store.read("img", path=found)


# ------------------------------------------------------------- what read() sees per format
# read() is a grayscale decode + ``> MASK_THRESHOLD`` for every file kind. These pin that
# contract for each kind a project can hold, so a faster decode path must keep it exactly:
# the alpha channel is ignored, colours go through the luma weights, 127 itself is off.
def _grid_bgr(colours) -> np.ndarray:
    """One 4x4 block per colour, side by side."""
    return np.concatenate([np.full((4, 4, len(c)), c, np.uint8) for c in colours], axis=1)


def _read_png(tmp_path, img) -> np.ndarray:
    store = MaskStore.for_image(str(tmp_path / "a.png"))
    os.makedirs(store.directory, exist_ok=True)
    assert cv2.imwrite(store.path_for("a"), img)
    return store.read("a")


def _blocks(on: np.ndarray) -> list[bool]:
    """The value of each 4x4 block (asserting the block is uniform)."""
    values = []
    for x in range(0, on.shape[1], 4):
        block = on[:, x : x + 4]
        assert block.all() or not block.any()
        values.append(bool(block[0, 0]))
    return values


def test_read_of_a_grey_model_mask_thresholds_at_127(tmp_path):
    grey = np.array([[0, 127, 128, 200, 255]], np.uint8).repeat(4, 0).repeat(4, 1)
    assert _blocks(_read_png(tmp_path, grey)) == [False, False, True, True, True]


def test_the_threshold_is_127():
    assert MASK_THRESHOLD == 127


def test_measure_and_the_mask_store_agree_on_a_grey_mask(tmp_path):
    img = tmp_path / "a.jpg"
    grey = np.arange(256, dtype=np.uint8).reshape(16, 16)  # 0..255 once each
    store = MaskStore.for_image(str(img))
    os.makedirs(store.directory)
    assert cv2.imwrite(store.path_for("a"), grey)
    measured = np.count_nonzero(metrics.load_mask(store.path_for("a")) == 255)
    assert measured == np.count_nonzero(store.read("a")) == 128


def test_read_of_rgba_ignores_the_alpha_channel(tmp_path):
    bgra = _grid_bgr(
        [(255, 255, 255, 255), (255, 255, 255, 128), (255, 255, 255, 0), (0, 0, 0, 255)]
    )
    assert _blocks(_read_png(tmp_path, bgra)) == [True, True, True, False]


@pytest.mark.parametrize("alpha", [None, 255, 77])
def test_read_of_coloured_pixels_uses_their_luma(tmp_path, alpha):
    colours = [
        (255, 255, 255),  # white: on
        (0, 255, 0),  # pure green (BGR), luma about 150: on
        (255, 255, 180),  # pale cyan-ish, luma about 223: on
        (201, 201, 201),  # light grey: on
        (200, 200, 200),  # grey 200 (was the threshold before O10): on
        (128, 128, 128),  # just over the threshold: on
        (127, 127, 127),  # exactly the threshold: off
        (0, 0, 255),  # pure red, luma about 76: off
        (255, 0, 0),  # pure blue, luma about 29: off (its first channel alone would be on)
    ]
    if alpha is not None:
        colours = [(*c, alpha) for c in colours]
    got = _blocks(_read_png(tmp_path, _grid_bgr(colours)))
    assert got == [True, True, True, True, True, True, False, False, False]


def test_read_of_a_legacy_jpg_mask(tmp_path):
    on = np.zeros((48, 64), bool)
    on[8:40, 16:48] = True
    store = MaskStore.for_image(str(tmp_path / "a.png"))
    os.makedirs(store.directory)
    path = os.path.join(store.directory, "a.jpg")
    assert cv2.imwrite(path, on.astype(np.uint8) * 255, [cv2.IMWRITE_JPEG_QUALITY, 95])
    assert store.find("a") == path
    assert np.array_equal(store.read("a"), cv2.imread(path, cv2.IMREAD_GRAYSCALE) > MASK_THRESHOLD)
    assert np.array_equal(store.read("a")[12:36, 20:44], on[12:36, 20:44])  # inside the edges


# --------------------------------------------------------------- the area Measure writes
# The field of view of the default calibration: 18 x 13.5 mm.
FOV_MM = (18.0, 13.5)


def test_the_measured_area_of_a_saved_mask_is_its_pixels_times_the_pixel_area(tmp_path):
    on = np.zeros((480, 640), bool)
    on[100:150, 200:300] = True  # 50 x 100 = 5000 pixels
    path = str(tmp_path / "gt" / "mask" / "T001_L044_2024.07.18.png")
    write_mask_file(path, on)
    calibration = Calibration(*FOV_MM)

    row = metrics.area_row("T001_L044_2024.07.18.jpg", path, calibration=calibration)

    # One pixel is (18 / 640) x (13.5 / 480) = 0.028125 x 0.028125 mm.
    assert metrics.root_area_mm2(metrics.load_mask(path), calibration=calibration) == (
        pytest.approx(5000 * 0.028125 * 0.028125)  # 3.955078125 mm²
    )
    assert row["Area (mm²)"] == 3.96  # the CSV keeps 2 decimals


def test_the_measured_area_follows_every_saved_pixel(tmp_path):
    rng = np.random.default_rng(23)
    calibration = Calibration(*FOV_MM)
    path = str(tmp_path / "gt" / "mask" / "a.png")
    for _ in range(25):
        h, w = (int(v) for v in rng.integers(1, 400, 2))
        on = rng.random((h, w)) < rng.random()
        write_mask_file(path, on, durable=False)
        pixels = sum(int(v) for v in on.ravel())  # counted without numpy's count_nonzero
        per_pixel = (FOV_MM[0] / w) * (FOV_MM[1] / h)

        area = metrics.root_area_mm2(metrics.load_mask(path), calibration=calibration)

        assert area == pytest.approx(pixels * per_pixel, rel=1e-12, abs=0.0)
