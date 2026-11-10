"""``MaskStore.read`` reads every mask file exactly as the old read did.

The old read (before any fast path) was a grayscale decode thresholded at
``MASK_THRESHOLD``, and ``MaskFileError`` where the decode returned None. For an RGBA
PNG libpng strips the alpha channel (it is not composited) and turns RGB into grey with
the luma weights; a pixel with R == G == B keeps its value. An editor-saved mask (8-bit
RGBA, R == G == B) and a model mask (8-bit grey) take a faster path that must give
the same grey plane.

These tests write many kinds of mask file (grey, RGBA with every alpha pattern, colour,
RGB, 16-bit, palette, every PNG row filter, odd sizes) and compare the store with a copy
of the old read.
"""

from __future__ import annotations

import os

import cv2
import numpy as np
import pytest

import app.masks.store as store_module
from app.io_utils import imread_unicode
from app.masks import MASK_THRESHOLD, MaskFileError, MaskStore, encode_mask_png
from tests.mask_corpus_support import _FILTERS, _strokes
from tests.mask_corpus_support import RGBA_CASES as CASES


def _old_read(path: str) -> np.ndarray:
    """``MaskStore.read`` before the fast path, copied as the reference."""
    gray = imread_unicode(path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise MaskFileError(f"cannot decode {path}")
    return gray > MASK_THRESHOLD


def _write(tmp_path, data: bytes) -> tuple[MaskStore, str]:
    store = MaskStore.for_image(str(tmp_path / "a.png"))
    os.makedirs(store.directory, exist_ok=True)
    path = store.path_for("a")
    with open(path, "wb") as fh:
        fh.write(data)
    return store, path


def test_the_filter_cases_really_use_different_filters():
    filtered = {CASES[f"rgba_filter_{name}"][0] for name in _FILTERS}
    assert len(filtered) == len(_FILTERS)


@pytest.mark.parametrize("name", sorted(CASES))
def test_read_is_the_old_read(tmp_path, name):
    store, path = _write(tmp_path, CASES[name][0])
    try:
        expected = _old_read(path)
    except MaskFileError:
        with pytest.raises(MaskFileError):
            store.read("a")
        return
    got = store.read("a")
    assert got.dtype == bool and got.shape == expected.shape
    assert np.array_equal(got, expected)


@pytest.mark.parametrize("name", sorted(CASES))
def test_fast_grey_plane_is_the_reference_grey_plane(name):
    """Stronger than the thresholded mask: the fast plane equals the decoded grey plane."""
    data, fast = CASES[name]
    plane = store_module._decode_plain_png(data)
    assert (plane is not None) == fast
    if plane is not None:
        reference = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
        assert plane.dtype == np.uint8 and plane.shape == reference.shape
        assert np.array_equal(plane, reference)


def test_random_editor_masks_read_back_exactly(tmp_path):
    rng = np.random.default_rng(11)
    for i in range(40):
        h, w = (int(v) for v in rng.integers(1, 300, 2))
        on = _strokes(h, w, i) ^ (rng.random((h, w)) < 0.01)
        store, path = _write(tmp_path, encode_mask_png(on))
        got = store.read("a")
        assert np.array_equal(got, on)
        assert np.array_equal(got, _old_read(path))
