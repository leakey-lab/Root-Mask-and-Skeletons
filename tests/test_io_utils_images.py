"""``app.io_utils`` image I/O: Unicode-safe read and write, and the one decoder of file bytes.

Promises: ``imread_unicode`` gives ``cv2.imread``'s pixels on any path (non-ASCII too) and
None for a missing, empty, undecodable file or a folder; ``imdecode_bytes`` is the same decode
on bytes already read (bytes, bytearray, memoryview, ``np.fromfile`` array), flags required;
``imwrite_unicode`` encodes by the extension, passes encoder flags, writes atomically, and
raises ``ValueError`` (no file) when it cannot encode. Measure, the GT check and the mask
readers decode through it, so they see the same pixels (O10).
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.inference import measure_sources, metrics
from app.io_utils import (
    imdecode_bytes,
    imread_unicode,
    imwrite_unicode,
)
from app.masks import store

UNICODE_DIRNAME = "Donn\u00e9es_\u6839\u7cfb"


def _leftovers(directory) -> list[str]:
    return sorted(os.listdir(directory))


def _sample_image() -> np.ndarray:
    img = np.zeros((48, 64, 3), dtype=np.uint8)
    cv2.line(img, (2, 2), (60, 40), (255, 255, 255), 1)
    img[5:9, 5:9] = (10, 200, 30)
    return img


# ------------------------------------------------------------------ read and write


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


# ------------------------------------------------------------------ the decoder of file bytes (O10)


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "docs" / "examples"
FLAGS = {
    "default": None,
    "color": cv2.IMREAD_COLOR,
    "grayscale": cv2.IMREAD_GRAYSCALE,
    "unchanged": cv2.IMREAD_UNCHANGED,
    "anydepth": cv2.IMREAD_ANYDEPTH,
}


def _encoded(img: np.ndarray, ext: str) -> bytes:
    ok, buf = cv2.imencode(ext, img)
    assert ok
    return buf.tobytes()


def _first(pattern: str) -> Path:
    return sorted(EXAMPLES.glob(pattern))[0]


def _cases() -> dict[str, bytes]:
    """One file of each kind of input: the formats the app reads and writes, broken files,
    and one real photo, mask and skeleton of ``docs/examples/``."""
    rng = np.random.default_rng(11)
    grey = rng.integers(0, 256, (37, 53), dtype=np.uint8)
    colour = rng.integers(0, 256, (29, 41, 3), dtype=np.uint8)
    rgba = rng.integers(0, 256, (23, 31, 4), dtype=np.uint8)
    grey16 = rng.integers(0, 1 << 16, (17, 19), dtype=np.uint16)
    png = _encoded(colour, ".png")
    return {
        "grey_png": _encoded(grey, ".png"),
        "colour_png": png,
        "rgba_png": _encoded(rgba, ".png"),
        "grey16_png": _encoded(grey16, ".png"),
        "jpeg": _encoded(colour, ".jpg"),
        "bmp": _encoded(colour, ".bmp"),
        "editor_mask": store.encode_mask_png(grey > 127),
        "empty": b"",
        "one_byte": b"\x89",
        "png_signature_only": png[:8],
        "truncated_png": png[: len(png) // 2],
        "garbage": rng.integers(0, 256, 300, dtype=np.uint8).tobytes(),
        "example_photo": _first("*.jpg").read_bytes(),
        "example_mask": _first("mask/*.png").read_bytes(),
        "example_skeleton": _first("skeletons/*.png").read_bytes(),
    }


CASES = _cases()


def _same(a: np.ndarray | None, b: np.ndarray | None) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b)


def _cv2_read(path: Path, flags: int) -> np.ndarray | None:
    """The oracle: OpenCV's own file reader (``tmp_path`` is an ASCII path, where it works)."""
    assert str(path).isascii(), "cv2.imread needs an ASCII path to be the oracle"
    return cv2.imread(str(path), flags)


@pytest.mark.parametrize("flag_name", sorted(FLAGS))
@pytest.mark.parametrize("name", sorted(CASES))
def test_every_reader_decodes_as_cv2_imread(tmp_path, name, flag_name):
    data = CASES[name]
    flags = FLAGS[flag_name]
    path = tmp_path / "f.bin"
    path.write_bytes(data)
    expected = _cv2_read(path, cv2.IMREAD_COLOR if flags is None else flags)

    assert _same(imread_unicode(str(path), flags), expected)
    if flags is not None:
        as_array = np.fromfile(str(path), np.uint8)
        for given in (data, bytearray(data), memoryview(data), as_array):
            assert _same(imdecode_bytes(given, flags), expected)


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_grey_readers_of_measure_and_the_masks_decode_alike(tmp_path, name):
    data = CASES[name]
    path = tmp_path / "f.png"
    path.write_bytes(data)
    expected = _cv2_read(path, cv2.IMREAD_GRAYSCALE)

    # metrics._read_grey is private: the only way to see Measure's decode before it thresholds.
    assert _same(metrics._read_grey(str(path), data), expected)
    assert _same(metrics._read_grey(str(path), None), expected)
    on = store.read_mask_file(str(path))
    assert _same(on, None if expected is None else expected > store.MASK_THRESHOLD)


def test_the_flags_have_no_default():
    with pytest.raises(TypeError):
        imdecode_bytes(CASES["grey_png"])  # type: ignore[call-arg]


@pytest.mark.parametrize("name", ["grey_png", "editor_mask", "truncated_png", "empty"])
def test_a_gt_file_is_judged_on_the_reference_decode(tmp_path, name):
    data = CASES[name]
    path = tmp_path / "a.png"
    path.write_bytes(data)
    expected = _cv2_read(path, cv2.IMREAD_GRAYSCALE)
    photo = tmp_path / "photo.png"
    shape = (37, 53) if expected is None else expected.shape
    photo.write_bytes(_encoded(np.zeros(shape, np.uint8), ".png"))

    decision = measure_sources.decide_area_source("a", None, str(path), str(photo))

    assert decision.reason == ("unreadable" if expected is None else "")
    assert (decision.data is None) == (expected is None)


def test_imwrite_unicode_writes_png_to_a_path_without_extension(tmp_path):
    path = tmp_path / "no_extension"

    imwrite_unicode(path, _sample_image())

    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert np.array_equal(imread_unicode(path), _sample_image())


def test_an_encoder_that_reports_failure_raises_and_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(cv2, "imencode", lambda ext, img, params: (False, None))

    with pytest.raises(ValueError, match="Cannot encode image as .png"):
        imwrite_unicode(tmp_path / "a.png", _sample_image())

    assert _leftovers(tmp_path) == []
