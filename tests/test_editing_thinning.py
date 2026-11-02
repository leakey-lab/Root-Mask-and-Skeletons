"""The skeleton thinning helpers of ``app.editing.thinning`` (Qt-free, W1-T3, O2).

``thin_raster`` (a skeleton file as Correct opens it) and the lazy scikit-image import moved
here from ``app.gui.skeleton_graph_model``, which imports them back under the same names.
``is_generated_as_opened`` says whether a skeleton is the generated one as Correct opens it:
a skeleton Save writes a correction only when it is not.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

import app.gui.skeleton_graph_model as sgm
from app.editing import thinning
from app.editing.thinning import is_generated_as_opened, thin_raster

ROOT = Path(__file__).resolve().parents[1]
PHOTO = (640, 480)  # (w, h)


def _line(row: int) -> np.ndarray:
    """A generated skeleton raster at SKELETON_OUTPUT_SIZE (341 x 256): one line on ``row``."""
    gray = np.zeros((256, 341), np.uint8)
    gray[row, 30:300] = 255
    return gray


def _generated(tmp_path) -> tuple[str, np.ndarray]:
    """The generated skeleton file of the tests (one line on row 100) and its raster."""
    gray = _line(100)
    path = str(tmp_path / "a_fake.png")
    cv2.imwrite(path, gray)
    return path, gray


def test_the_thinning_helpers_live_in_app_editing_and_load_no_qt():
    probe = (
        "import sys, app.editing.thinning; "
        "print(sorted(m for m in sys.modules if m == 'PyQt6' or m.startswith('PyQt6.') "
        "or m == 'skimage' or m.startswith('skimage.')))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], cwd=ROOT, capture_output=True, text=True, timeout=60
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"


def test_skeleton_graph_model_reexports_the_same_functions():
    assert sgm.thin_raster is thinning.thin_raster
    assert sgm.prewarm_skeletonize is thinning.prewarm_skeletonize
    assert sgm._skeletonize_uint8 is thinning._skeletonize_uint8
    assert sgm._ensure_uint8_binary is thinning._ensure_uint8_binary


def test_is_generated_as_opened(tmp_path):
    path, gray = _generated(tmp_path)
    opened = thin_raster(gray, PHOTO)
    other = thin_raster(_line(40), PHOTO)
    empty = np.zeros((PHOTO[1], PHOTO[0]), np.uint8)

    assert opened.any()
    assert is_generated_as_opened(opened, path, PHOTO)
    assert not is_generated_as_opened(other, path, PHOTO)
    assert is_generated_as_opened(empty, None, PHOTO)
    assert not is_generated_as_opened(empty, path, PHOTO)  # emptied: a real correction
    assert is_generated_as_opened(empty, str(tmp_path / "missing.png"), PHOTO)
    assert not is_generated_as_opened(opened, str(tmp_path / "missing.png"), PHOTO)


def test_an_unreadable_generated_file_counts_as_empty(tmp_path):
    path = tmp_path / "broken_fake.png"
    path.write_bytes(b"not a png")
    empty = np.zeros((PHOTO[1], PHOTO[0]), np.uint8)

    assert is_generated_as_opened(empty, str(path), PHOTO)
    assert not is_generated_as_opened(empty + 255, str(path), PHOTO)


def test_the_values_of_the_mask_do_not_matter_only_where_it_is_set(tmp_path):
    path, gray = _generated(tmp_path)
    opened = thin_raster(gray, PHOTO)

    assert is_generated_as_opened((opened > 0).astype(np.uint8), path, PHOTO)


def _reference_skeletonize_uint8(mask: np.ndarray) -> np.ndarray:
    """``_skeletonize_uint8`` as it was before the lean form (2026-10-26): the 0/255 copy of
    ``_ensure_uint8_binary``, then two passes back to 0/255."""
    mask = thinning._ensure_uint8_binary(mask)
    skel_bool = thinning._skeletonize_fn()(mask > 0)
    return skel_bool.astype(np.uint8) * 255


def _layouts(plane: np.ndarray):
    yield plane
    yield np.asfortranarray(plane)
    yield plane[::2, 1::3]
    yield plane.T
    yield plane[::-1, ::-1]


def test_the_skeletonize_wrapper_gives_the_reference_bytes_for_any_input():
    rng = np.random.default_rng(20261026)
    planes = []
    for dtype in (np.uint8, np.int8, np.uint16, np.int16, np.int32, np.int64):
        values = rng.integers(-300, 600, (37, 53)) * (rng.random((37, 53)) < 0.4)
        planes.append(values.astype(dtype))
    for dtype in (np.float32, np.float64):
        values = rng.normal(0.0, 200.0, (37, 53)) * (rng.random((37, 53)) < 0.4)
        values[0, :4] = (0.5, -0.5, 255.9, 256.0)
        planes.append(values.astype(dtype))
    planes.append(rng.random((37, 53)) < 0.3)  # bool
    blob = np.zeros((60, 80), np.uint8)
    cv2.circle(blob, (40, 30), 20, 255, -1)  # a thick blob: thinning removes pixels
    cv2.line(blob, (0, 59), (79, 0), 7, 3)  # values other than 255
    planes.append(blob)
    compared = 0
    with np.errstate(invalid="ignore"):
        for plane in planes:
            for mask in _layouts(plane):
                want = _reference_skeletonize_uint8(mask)
                got = thinning._skeletonize_uint8(mask)
                assert got.dtype == want.dtype == np.uint8 and got.shape == want.shape
                assert np.array_equal(got, want), (plane.dtype, mask.strides)
                assert got.flags.c_contiguous and got.flags.writeable and got.flags.owndata
                compared += 1
    assert compared == len(planes) * 5


def test_the_skeletonize_wrapper_refuses_what_is_not_a_plane():
    for bad in (np.zeros(5, np.uint8), np.zeros((4, 5, 3), np.uint8), np.uint8(1), np.zeros(())):
        for fn in (thinning._skeletonize_uint8, _reference_skeletonize_uint8):
            try:
                fn(np.asarray(bad))
            except ValueError as exc:
                assert str(exc) == "Expected a 2D binary mask."
            else:
                raise AssertionError(f"{fn.__name__} took a {np.asarray(bad).shape} array")


# ------------------------------------------------------------- the bilinear upscale
#: (source (w, h), target (w, h)): a prediction to photo sizes, downscales, the exact 2x
#: shrink (OpenCV serves it from INTER_AREA), odd and 1-px sizes
_RESIZE_SIZES = [
    ((341, 256), (640, 480)),
    ((341, 256), (2048, 1536)),
    ((640, 480), (341, 256)),
    ((682, 512), (341, 256)),
    ((100, 80), (37, 53)),
    ((13, 7), (5, 3)),
    ((5, 3), (13, 7)),
    ((1, 1), (9, 9)),
    ((300, 1), (77, 5)),
    ((640, 480), (639, 481)),
]
#: sha256 of resize_linear_u8 over _resize_cases(): recorded on x86, where it equals
#: cv2.resize, and the same under QEMU aarch64 (2026-10-29, A7)
_RESIZE_DIGEST = "dda0a814447e3128a745293399ad95c06bc5ab66cb377afbe5b9a09168effd51"


def _resize_cases():
    rng = np.random.default_rng(29)
    for (sw, sh), size in _RESIZE_SIZES:
        yield rng.integers(0, 256, (sh, sw), dtype=np.uint8), size


def _resize_goes_to_an_arm_hal() -> bool:
    """True for OpenCV builds with the carotene or KleidiCV HAL (the arm64 wheels), whose
    cv2.resize rounds unlike OpenCV's own code."""
    hal = [line for line in cv2.getBuildInformation().splitlines() if "Custom HAL:" in line]
    return any(name in line for line in hal for name in ("carotene", "KleidiCV"))


def test_the_upscale_gives_the_same_pixels_on_every_platform():
    """thin_raster's resize is integer numpy: the digest recorded on x86 holds on arm64 too,
    where cv2.resize gave other pixels (macOS CI run 40970595195)."""
    digest = hashlib.sha256()
    for img, size in _resize_cases():
        out = thinning.resize_linear_u8(img, size)
        assert out.dtype == np.uint8 and out.shape == (size[1], size[0])
        digest.update(out.tobytes())
    assert digest.hexdigest() == _RESIZE_DIGEST


@pytest.mark.skipif(
    _resize_goes_to_an_arm_hal(), reason="cv2.resize goes to an arm64 HAL here: why the port exists"
)
def test_the_upscale_equals_opencvs_bilinear_resize():
    rng = np.random.default_rng(123)
    cases = list(_resize_cases())
    for i in range(80):
        sw, sh = int(rng.integers(1, 400)), int(rng.integers(1, 400))
        if i % 4 == 0:  # the exact 2x shrink
            sw, sh = 2 * max(1, sw // 2), 2 * max(1, sh // 2)
            size = (sw // 2, sh // 2)
        else:
            size = (int(rng.integers(1, 900)), int(rng.integers(1, 900)))
        cases.append((rng.integers(0, 256, (sh, sw), dtype=np.uint8), size))
    for img, size in cases:
        want = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
        assert np.array_equal(thinning.resize_linear_u8(img, size), want), (img.shape, size)
