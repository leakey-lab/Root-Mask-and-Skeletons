"""Every reader of a mask file sees the reference decode, bit for bit (``app.masks.store``).

The reference decode is ``imread_unicode(path, IMREAD_GRAYSCALE)`` thresholded at
``MASK_THRESHOLD`` (127 is off, 128 on), or "unreadable" where it returns None. For an RGBA
PNG libpng strips the alpha channel (it does not composite it) and turns RGB into grey with
the luma weights, so a pixel with R == G == B keeps its value.

``read_mask_file`` decodes the bytes it read once. A plain 8-bit grey PNG (a model mask) and
a plain 8-bit RGBA PNG whose R == G == B (an editor mask) take an exact Pillow fast path; any
other file goes to the reference decode. Pillow and libpng disagree on some broken files
(Pillow accepts a zlib stream that ends early or never ends, and ignores parts of the IHDR),
so the fast path must refuse those.

The differential runs over ``tests/mask_corpus_support.CORPUS``: about 200 distinct files
(grey, RGBA with every alpha pattern, colour, RGB, 16-bit, palette, interlaced, every PNG row
filter and several zlib settings from OpenCV and Pillow, odd sizes, broken chunks and
streams, a legacy JPEG). The corpus also guards upgrades of the decoders (Pillow, zlib-ng,
libpng in OpenCV): a new version that decodes one of these files differently fails here,
whatever this repository's code does. Each file is checked through every reader: the fast
path's routing and grey plane, ``read_mask_file``, ``MaskStore.read``, Correct's GT read
(``mask_files._read_gt``) and Measure's area reader (``metrics.load_mask``).
"""

from __future__ import annotations

import os

import cv2
import numpy as np
import pytest
from PIL import Image

import app.masks.store as store_module
from app.gui.correct import mask_files
from app.inference import metrics
from app.io_utils import imread_unicode
from app.masks import MaskFileError, MaskStore, encode_mask_png, read_mask_file
from tests.mask_corpus_support import _FILTERS, CORPUS, RGBA_CASES, _pil_png, _strokes

#: The owner's binarisation rule (O10), written out: the reference shares no constant with the
#: code under test, so a changed ``MASK_THRESHOLD`` fails here too.
REFERENCE_THRESHOLD = 127


def _write(tmp_path, data: bytes, name: str = "a.png") -> tuple[MaskStore, str]:
    """``data`` as the prediction ``mask/<name>`` of image ``a.png``."""
    store = MaskStore.for_image(str(tmp_path / "a.png"))
    os.makedirs(store.directory, exist_ok=True)
    path = os.path.join(store.directory, name)
    with open(path, "wb") as fh:
        fh.write(data)
    return store, path


def _reference_read(path: str) -> np.ndarray | None:
    """The reference decode: the file's grey plane above the threshold (None: unreadable)."""
    grey = imread_unicode(path, cv2.IMREAD_GRAYSCALE)
    return None if grey is None else grey > REFERENCE_THRESHOLD


@pytest.mark.parametrize("name", sorted(CORPUS))
def test_every_mask_file_reads_as_the_reference_decode(tmp_path, name):
    data, fast = CORPUS[name]
    store, path = _write(tmp_path, data, "a.jpg" if name == "editor:legacy_jpg" else "a.png")
    grey = imread_unicode(path, cv2.IMREAD_GRAYSCALE)

    # The fast path takes exactly the plain 8-bit grey and grey RGBA PNGs, and then its grey
    # plane (before the threshold) is the reference plane.
    plane = store_module._decode_plain_png(data)
    assert (plane is not None) == fast, "the fast path took the wrong files"
    if plane is not None:
        assert plane.dtype == np.uint8 and np.array_equal(plane, grey)

    if grey is None:  # unreadable: reported by every reader, never read as an empty mask
        assert read_mask_file(path) is None
        with pytest.raises(MaskFileError):
            store.read("a")
        assert mask_files._read_gt(path, (61, 47)) == (None, "unreadable")
        with pytest.raises(FileNotFoundError):
            metrics.load_mask(path)
        return
    expected = grey > REFERENCE_THRESHOLD
    height, width = grey.shape
    got = read_mask_file(path)
    assert got.dtype == bool and got.shape == expected.shape
    assert np.array_equal(got, expected)
    assert np.array_equal(store.read("a"), expected)
    gt_plane, why = mask_files._read_gt(path, (width, height))
    assert why == "" and gt_plane.dtype == np.uint8
    assert np.array_equal(gt_plane, expected.astype(np.uint8))
    assert mask_files._read_gt(path, (width + 1, height)) == (None, "size")
    # Measure counts the same pixels as root (root_areas.csv).
    assert np.array_equal(metrics.load_mask(path) == 255, expected)


def test_the_filter_cases_really_use_different_filters():
    filtered = {RGBA_CASES[f"rgba_filter_{name}"][0] for name in _FILTERS}
    assert len(filtered) == len(_FILTERS)


def test_random_model_masks_read_back_exactly(tmp_path):
    rng = np.random.default_rng(17)
    for _ in range(40):
        h, w = (int(v) for v in rng.integers(1, 300, 2))
        grey = (rng.random((h, w)) < 0.3).astype(np.uint8) * 255
        grey[rng.random((h, w)) < 0.05] = rng.integers(190, 211)
        data = _pil_png(Image.fromarray(grey, "L"), compress_level=int(rng.integers(0, 10)))
        store, path = _write(tmp_path, data)
        assert store_module._decode_plain_png(data) is not None
        assert np.array_equal(store.read("a"), grey > REFERENCE_THRESHOLD)
        assert np.array_equal(store.read("a"), _reference_read(path))


def test_random_editor_masks_read_back_exactly(tmp_path):
    rng = np.random.default_rng(11)
    for i in range(40):
        h, w = (int(v) for v in rng.integers(1, 300, 2))
        on = _strokes(h, w, i) ^ (rng.random((h, w)) < 0.01)
        store, path = _write(tmp_path, encode_mask_png(on))
        got = store.read("a")
        assert np.array_equal(got, on)
        assert np.array_equal(got, _reference_read(path))
